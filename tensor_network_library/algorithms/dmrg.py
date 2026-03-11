"""
DMRG (Density Matrix Renormalization Group) algorithm.

Uses Schollwöck's mixed-canonical form: sweep left-to-right and right-to-left,
solving two-site optimization problems at each step.

Key steps:
1. Start with an MPS in left-canonical form (center=0).
2. Sweep right: optimize two-site object (sites i, i+1), left-normalize, move center.
3. At rightmost site, sweep left: optimize two-site object, right-normalize, move center.
4. Repeat until convergence.
"""

from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
from scipy.linalg import eigh

from ..core.tensor import Tensor
from ..core.mps import MPS
from ..core.mpo import MPO
from ..core.policy import TruncationPolicy
from ..core.index import Index


# --------------------------
# Output
# --------------------------

class DMRGResult:
    """ A class that sets the standard for the results of a DMRG ground state search."""

    def __init__(
            self,
            mps: MPS,
            energy: float,
            iterations: int,
            energy_history: List[float],
            entanglement_history: List[float],
            ):
        
        self.mps = mps
        self.energy = energy
        self.iterations = iterations
        self.energy_history = energy_history
        self.entanglement_history = entanglement_history


# ----------------------------
# DMRG algorithm
# ----------------------------

class DMRG:
    """
    DMRG ground state finder.
    """

    def __init__(self, 
                 mps: MPS, 
                 mpo: MPO, 
                 truncation_policy: Optional[TruncationPolicy] = None,
                 ):
        
        """ 
        Initialize the environment for a DMRG calculation.
        
        Args:
            mps: MPS to optimize
            mpo: MPO Hamiltonian
            truncation_policy: Policy for SVD truncation (default: keep all singular values)
        """

        assert mps.L == mpo.L, "MPS and MPO must have the same length."
        assert mps.physical_dims == mpo.d, "MPS and MPO must have the same physical dimesnion."

        self.mps = mps
        self.mpo = mpo
        self.truncation_policy = truncation_policy or TruncationPolicy()

        self._init_environment()

    
    def _init_environments(self) -> None:
        """
        Initialize left and right effective Hamiltonian environments.
        
        L[i] = left contraction of sites [0:i+1]
        R[i] = right contraction of sites [i:L]
        """
        self.L_envs = [None] * (self.mps.L + 1)  # L_envs[i] covers sites [0:i]
        self.R_envs = [None] * (self.mps.L + 1)  # R_envs[i] covers sites [i:L]
        
        # Initialize boundaries
        self.L_envs[0] = self._make_trivial_left_env()
        self.R_envs[self.mps.L] = self._make_trivial_right_env()

    def _make_trivial_left_env(self) -> Tensor:
        """Create trivial left environment (left boundary)."""
        # Shape: (1, 1, 1) = (D_mpo_left=1, chi_mps_left=1, chi_mps_left=1)
        data = np.ones((1, 1, 1), dtype=np.complex128)
        inds = [
            Index(dim=1, name="E_L_left_mpo"),
            Index(dim=1, name="E_L_left_mps"),
            Index(dim=1, name="E_L_right_mps"),
        ]
        return Tensor(data, inds=inds)

    def _make_trivial_right_env(self) -> Tensor:
        """Create trivial right environment (right boundary)."""
        # Shape: (1, 1, 1) = (D_mpo_right=1, chi_mps_right=1, chi_mps_right=1)
        data = np.ones((1, 1, 1), dtype=np.complex128)
        inds = [
            Index(dim=1, name="E_R_left_mpo"),
            Index(dim=1, name="E_R_left_mps"),
            Index(dim=1, name="E_R_right_mps"),
        ]
        return Tensor(data, inds=inds)

    def _expand_left_env(self, i: int) -> None:
        """
        Compute L_envs[i+1] from L_envs[i] by contracting site i.
        
        L[i+1] = contract(L[i], mps[i], mpo[i])
        """
        if i < 0 or i >= self.mps.L:
            return
        
        if self.L_envs[i + 1] is not None:
            return  # Already computed
        
        # Simplified contraction (full implementation would use proper tensor indices)
        # For now, assume left environments are 3-legged: (mpo_bond, mps_left, mps_left_dag)
        L = self.L_envs[i]
        psi = self.mps.tensors[i]  # shape (chi_l, d, chi_r)
        W = self.mpo.tensors[i]  # shape (D_l, d_in, d_out, D_r)
        
        # Contraction (simplified; real implementation uses einsum with careful index management)
        # Result should have shape (D_r, chi_r, chi_r)
        L_new_data = np.ones(
            (W.shape[-1], psi.shape[-1], psi.shape[-1]), dtype=np.complex128
        )
        
        L_new = Tensor(L_new_data, inds=self.L_envs[i].inds.copy())
        self.L_envs[i + 1] = L_new

    def _expand_right_env(self, i: int) -> None:
        """
        Compute R_envs[i-1] from R_envs[i] by contracting site i.
        
        R[i-1] = contract(mps[i], mpo[i], R[i])
        """
        if i <= 0 or i > self.mps.L:
            return
        
        if self.R_envs[i - 1] is not None:
            return  # Already computed
        
        R = self.R_envs[i]
        psi = self.mps.tensors[i - 1]  # site i-1
        W = self.mpo.tensors[i - 1]  # MPO at site i-1
        
        # Result should have shape (D_l, chi_l, chi_l)
        R_new_data = np.ones(
            (W.shape[0], psi.shape[0], psi.shape[0]), dtype=np.complex128
        )
        
        R_new = Tensor(R_new_data, inds=self.R_envs[i].inds.copy())
        self.R_envs[i - 1] = R_new


    def _two_site_ham(self, site_l: int) -> np.ndarray:
        """
        Build effective two-site Hamiltonian matrix for sites (site_l, site_l+1).
        
        Returns flattened matrix suitable for eigenvalue solver.
        """
        psi_l = self.mps.tensors[site_l]  # (chi_l, d, chi_mid)
        psi_r = self.mps.tensors[site_l + 1]  # (chi_mid, d, chi_r)
        
        # Effective size: (chi_l * d) * (d * chi_r)
        size = psi_l.shape[0] * psi_l.shape[1] * psi_r.shape[1] * psi_r.shape[2]
        
        # Build Hamiltonian: H_eff = L[site_l] @ W[site_l] @ W[site_l+1] @ R[site_l+2]
        # For simplicity, return identity-scaled matrix (placeholder)
        H_eff = np.eye(size, dtype=np.complex128)
        
        return H_eff

    def optimize_two_site(self, site_l: int) -> float:
        """
        Optimize two-site object at positions (site_l, site_l+1).
        
        Solves the eigenvalue problem, then updates MPS using SVD.
        
        Returns:
            Energy eigenvalue.
        """
        H_eff = self._two_site_ham(site_l)
        
        # Solve eigenvalue problem
        _, psi_opt = eigsh(H_eff, k=1, which='SA')
        psi_opt = psi_opt[:, 0]
        
        # Reshape back to two-site tensor
        shape_two_site = (
            self.mps.tensors[site_l].shape[0],
            self.mps.tensors[site_l].shape[1],
            self.mps.tensors[site_l + 1].shape[1],
            self.mps.tensors[site_l + 1].shape[2],
        )
        psi_two_site = psi_opt.reshape(shape_two_site)
        
        # SVD to update MPS
        # Group (chi_l, d_l) on left, (d_r, chi_r) on right
        left_shape = (shape_two_site[0] * shape_two_site[1], shape_two_site[2] * shape_two_site[3])
        mat = psi_two_site.reshape(left_shape)
        
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        
        # Update tensors with truncation
        chi_new = min(len(S), self.mps.chi_list[site_l + 1])
        U = U[:, :chi_new]
        S = S[:chi_new]
        Vh = Vh[:chi_new, :]
        
        # Reshape and store
        self.mps.tensors[site_l] = Tensor(
            U.reshape(shape_two_site[0], shape_two_site[1], chi_new),
            inds=self.mps.tensors[site_l].inds,
        )
        
        # Absorb S into right tensor
        psi_r_new = (np.diag(S) @ Vh).reshape(chi_new, shape_two_site[2], shape_two_site[3])
        self.mps.tensors[site_l + 1] = Tensor(
            psi_r_new,
            inds=self.mps.tensors[site_l + 1].inds,
        )
        
        # Return energy
        energy = np.real(np.dot(psi_opt, H_eff @ psi_opt))
        return energy

    def sweep_right_to_left(self) -> float:
        """
        Perform one right-to-left sweep.
        
        Returns:
            Ground state energy estimate.
        """
        energies = []
        for site in range(self.mps.L - 2, -1, -1):
            energy = self.optimize_two_site(site)
            energies.append(energy)
            self.mps.set_center(site)
        
        return np.mean(energies) if energies else 0.0

    def sweep_left_to_right(self) -> float:
        """
        Perform one left-to-right sweep.
        
        Returns:
            Ground state energy estimate.
        """
        energies = []
        for site in range(self.mps.L - 1):
            energy = self.optimize_two_site(site)
            energies.append(energy)
            self.mps.set_center(site + 1)
        
        return np.mean(energies) if energies else 0.0

    def run(
        self,
        num_sweeps: int = 5,
        tolerance: float = 1e-6,
    ) -> DMRGResult:
        """
        Run DMRG optimization.
        
        Args:
            num_sweeps: Number of sweep iterations.
            tolerance: Energy convergence tolerance.
        
        Returns:
            DMRGResult with optimized MPS and energy history.
        """
        energy_history = []
        entanglement_history = []
        
        prev_energy = None
        converged = False
        
        for sweep_idx in range(num_sweeps):
            # Alternate sweep directions
            if sweep_idx % 2 == 0:
                energy = self.sweep_left_to_right()
            else:
                energy = self.sweep_right_to_left()
            
            energy_history.append(energy)
            
            # Compute entanglement at center
            ent = self.mps.entanglement_entropy(self.mps.center)
            entanglement_history.append(ent)
            
            # Check convergence
            if prev_energy is not None:
                if abs(energy - prev_energy) < tolerance:
                    converged = True
                    break
            prev_energy = energy
            
            print(f"Sweep {sweep_idx + 1}: E = {energy:.10f}, S = {ent:.6f}")
        
        return DMRGResult(
            mps=self.mps,
            energy=prev_energy or energy_history[-1],
            iterations=len(energy_history),
            energy_history=energy_history,
            entanglement_history=entanglement_history,
        )