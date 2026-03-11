"""
DMRG: Density Matrix Renormalization Group for ground-state search.

Implements both single-site (1S-DMRG) and two-site (2S-DMRG) update
schemes for 1D systems with open boundary conditions.

Algorithm overview (both variants):
    1. Initialize MPS in right-canonical form.
    2. Build right environment blocks R[i] for all i.
    3. Sweep left-to-right, then right-to-left (one full sweep = L->R + R->L).
       At each site (single-site) or bond (two-site):
         a. Build effective Hamiltonian from left env L[i], MPO W[i], right env R[i+1].
         b. Solve the local eigenvalue problem via scipy ARPACK (sparse eigensolver).
         c. Update site tensor(s) and canonicalize.
         d. Update the environment block.
    4. Repeat sweeps until energy convergence.

Single-site DMRG:
    - Updates one site tensor per step.
    - Bond dimension is fixed (controlled by initial MPS).
    - Fast but can get stuck in local minima without perturbation.

Two-site DMRG:
    - Merges two adjacent site tensors, solves the 2-site problem, then splits
      via SVD with optional truncation.
    - Bond dimension can grow adaptively.
    - Better convergence for critical/entangled systems.

User-facing interface:
    - Hamiltonian is provided as an MPO (built by hamiltonian/models.py builders).
    - Initial MPS is provided (or a random MPS is generated internally).
    - Environment can optionally be passed for centralized settings.

Reference:
    Schollwoeck, U. (2011). The density-matrix renormalization group in the
    age of matrix product states. Annals of Physics 326, 96-192.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

from ..core.mps import MPS
from ..core.mpo import MPO
from ..core.tensor import Tensor
from ..core.index import Index
from ..core.policy import TruncationPolicy
from ..core.env import Environment


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def _init_left_env(chi_mps: int, chi_mpo: int, dtype) -> np.ndarray:
    env = np.zeros((chi_mps, chi_mpo, chi_mps), dtype=dtype)
    env[0, 0, 0] = 1.0
    return env


def _init_right_env(chi_mps: int, chi_mpo: int, dtype) -> np.ndarray:
    env = np.zeros((chi_mps, chi_mpo, chi_mps), dtype=dtype)
    env[0, 0, 0] = 1.0
    return env


def _update_left_env(L_env: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Contract left environment with MPS site tensor A and MPO tensor W.
    L_env: (chi_mps, chi_mpo, chi_mps)
    A:     (chi_l, d, chi_r)
    W:     (chi_mpo_l, d, d, chi_mpo_r)
    -> new L_env: (chi_r, chi_mpo_r, chi_r)
    """
    tmp = np.tensordot(L_env, A.conj(), axes=([0], [0]))
    tmp = np.tensordot(tmp, W, axes=([0, 2], [0, 2]))
    tmp = np.tensordot(tmp, A, axes=([0, 2], [0, 1]))
    return tmp


def _update_right_env(R_env: np.ndarray, B: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Contract right environment with MPS site tensor B and MPO tensor W.
    R_env: (chi_mps, chi_mpo, chi_mps)
    B:     (chi_l, d, chi_r)
    W:     (chi_mpo_l, d, d, chi_mpo_r)
    -> new R_env: (chi_l, chi_mpo_l, chi_l)
    """
    tmp = np.tensordot(R_env, B.conj(), axes=([0], [2]))
    tmp = np.tensordot(tmp, W, axes=([0, 2], [3, 2]))
    tmp = np.tensordot(tmp, B, axes=([0, 2], [2, 1]))
    return tmp


def _build_right_envs(mps: MPS, mpo: MPO) -> List[np.ndarray]:
    """Build all right environment blocks R[0..L]."""
    L     = mps.L
    dtype = mps.tensors[0].data.dtype
    R_envs = [None] * (L + 1)
    R_envs[L] = _init_right_env(mps.bond_dims[-1], mpo.bond_dims[-1], dtype)
    for i in range(L - 1, -1, -1):
        R_envs[i] = _update_right_env(R_envs[i + 1], mps.tensors[i].data, mpo.tensors[i].data)
    return R_envs


# ---------------------------------------------------------------------------
# Effective Hamiltonian actions
# ---------------------------------------------------------------------------

def _heff_single_site(
    L_env: np.ndarray, R_env: np.ndarray, W: np.ndarray,
    v: np.ndarray, shape: Tuple[int, int, int],
) -> np.ndarray:
    chi_l, d, chi_r = shape
    M   = v.reshape(chi_l, d, chi_r)
    tmp = np.tensordot(L_env, M,   axes=([2], [0]))
    tmp = np.tensordot(tmp,   W,   axes=([1, 2], [0, 2]))
    tmp = np.tensordot(tmp,   R_env, axes=([1, 3], [0, 1]))
    return tmp.reshape(-1)


def _heff_two_site(
    L_env: np.ndarray, R_env: np.ndarray,
    W_l: np.ndarray, W_r: np.ndarray,
    v: np.ndarray, shape: Tuple[int, int, int, int],
) -> np.ndarray:
    chi_l, d_l, d_r, chi_r = shape
    theta = v.reshape(chi_l, d_l, d_r, chi_r)
    tmp = np.tensordot(L_env, theta, axes=([2], [0]))
    tmp = np.tensordot(tmp,   W_l,   axes=([1, 2], [0, 2]))
    tmp = np.tensordot(tmp,   W_r,   axes=([2, 4], [3, 0]))
    tmp = np.tensordot(tmp,   R_env, axes=([-1, 1], [1, 0]))
    return tmp.reshape(-1)


# ---------------------------------------------------------------------------
# DMRG
# ---------------------------------------------------------------------------

class DMRG:
    """
    DMRG ground-state solver for 1D systems with OBC.

    Args:
        mpo:        MPO representing the Hamiltonian.
        mps:        Initial MPS (modified in-place). If None, a random MPS
                    is generated from env / truncation settings.
        variant:    '1S' (single-site) or '2S' (two-site).
        env:        Optional Environment for centralized system parameters.
        truncation: TruncationPolicy for SVD truncation (two-site only).
                    Falls back to env.effective_truncation when not set.
        n_sweeps:   Maximum number of full sweeps.
        tol:        Energy convergence tolerance per sweep.
        n_krylov:   Krylov vectors for eigsh.

    Example::

        from tensor_network_library.hamiltonian.models import tfim_mpo
        from tensor_network_library.algorithms.dmrg import DMRG
        from tensor_network_library.core.env import Environment

        env = Environment.qubit_chain(L=10, chi_max=32)
        mpo = tfim_mpo(L=10, J=1.0, g=1.0)
        E0, mps = DMRG(mpo=mpo, variant='2S', env=env).run()
    """

    def __init__(
        self,
        mpo: MPO,
        mps: Optional[MPS]         = None,
        variant: str               = "2S",
        env: Optional[Environment] = None,
        truncation: Optional[TruncationPolicy] = None,
        n_sweeps: int  = 10,
        tol: float     = 1e-10,
        n_krylov: int  = 4,
    ):
        variant = variant.upper()
        if variant not in ("1S", "2S"):
            raise ValueError(f"variant must be '1S' or '2S', got {variant!r}")

        self.mpo      = mpo
        self.variant  = variant
        self.env      = env
        self.n_sweeps = n_sweeps
        self.tol      = tol
        self.n_krylov = n_krylov
        self.energies: List[float] = []

        # Resolve truncation policy
        if truncation is not None:
            self.truncation = truncation
        elif env is not None:
            self.truncation = env.effective_truncation
        else:
            self.truncation = None

        L, d, dtype = mpo.L, mpo.d, mpo.dtype

        if mps is not None:
            self.mps = mps
        else:
            chi_init = 2
            if env is not None and env.max_bond_dim:
                chi_init = min(env.max_bond_dim, d ** (L // 2))
            elif self.truncation is not None and self.truncation.max_bond_dim:
                chi_init = min(self.truncation.max_bond_dim, d ** (L // 2))
            self.mps = self._random_mps(L, d, chi_init, dtype)

        self.mps.normalize()
        self._right_canonicalize()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_mps(L: int, d: int, chi: int, dtype) -> MPS:
        rng = np.random.default_rng()
        bond_dims = [1] + [min(chi, d**min(i+1, L-i-1)) for i in range(L-1)] + [1]
        bond_dims = [max(1, b) for b in bond_dims]
        tensors = []
        for i in range(L):
            shape = (bond_dims[i], d, bond_dims[i+1])
            data  = rng.standard_normal(shape).astype(np.float64)
            if np.issubdtype(dtype, np.complexfloating):
                data = data + 1j * rng.standard_normal(shape)
            b_l = Index(dim=bond_dims[i],   name=f"rMPS_bond_{i}",   tags=frozenset({"bond"}))
            p   = Index(dim=d,              name=f"rMPS_phys_{i}",   tags=frozenset({"phys"}))
            b_r = Index(dim=bond_dims[i+1], name=f"rMPS_bond_{i+1}", tags=frozenset({"bond"}))
            tensors.append(Tensor(data.astype(dtype), indices=[b_l, p, b_r]))
        return MPS.from_tensors(tensors)

    def _right_canonicalize(self) -> None:
        L = self.mps.L
        for i in range(L - 1, 0, -1):
            B = self.mps.tensors[i].data
            chi_l, d, chi_r = B.shape
            Q, R = np.linalg.qr(B.reshape(chi_l, d * chi_r).T)
            chi_new = Q.shape[1]
            self.mps.tensors[i].data = Q.T.reshape(chi_new, d, chi_r)
            self.mps.tensors[i].indices[0] = Index(
                dim=chi_new,
                name=self.mps.tensors[i].indices[0].name,
                tags=self.mps.tensors[i].indices[0].tags,
            )
            A = self.mps.tensors[i-1].data
            self.mps.tensors[i-1].data = (
                A.reshape(A.shape[0] * A.shape[1], chi_l) @ R.T
            ).reshape(A.shape[0], A.shape[1], chi_new)
            self.mps.tensors[i-1].indices[2] = self.mps.tensors[i].indices[0]
        self.mps.bonds = (
            [t.indices[0] for t in self.mps.tensors]
            + [self.mps.tensors[-1].indices[2]]
        )

    # ------------------------------------------------------------------
    # Sweeps
    # ------------------------------------------------------------------

    def _sweep(self, L_envs, R_envs) -> float:
        return self._sweep_1s(L_envs, R_envs) if self.variant == "1S" else self._sweep_2s(L_envs, R_envs)

    def _sweep_1s(self, L_envs, R_envs) -> float:
        L = self.mps.L
        energy = 0.0
        for i in range(L - 1):
            M, W, shape = self.mps.tensors[i].data, self.mpo.tensors[i].data, self.mps.tensors[i].data.shape
            op = LinearOperator(
                (int(np.prod(shape)),) * 2,
                matvec=lambda v, _s=shape, _L=L_envs[i], _R=R_envs[i+1], _W=W:
                    _heff_single_site(_L, _R, _W, v, _s),
                dtype=M.dtype,
            )
            vals, vecs = eigsh(op, k=max(1, min(self.n_krylov, int(np.prod(shape)) - 1)),
                               which="SA", v0=M.reshape(-1), tol=0)
            energy  = float(vals[0])
            M_new   = vecs[:, 0].reshape(shape)
            chi_l, d, chi_r = shape
            Q, R_mat = np.linalg.qr(M_new.reshape(chi_l * d, chi_r))
            self.mps.tensors[i].data   = Q.reshape(chi_l, d, Q.shape[1])
            self.mps.tensors[i+1].data = np.tensordot(R_mat, self.mps.tensors[i+1].data, axes=([1], [0]))
            L_envs[i+1] = _update_left_env(L_envs[i], self.mps.tensors[i].data, W)

        for i in range(L - 1, 0, -1):
            M, W, shape = self.mps.tensors[i].data, self.mpo.tensors[i].data, self.mps.tensors[i].data.shape
            op = LinearOperator(
                (int(np.prod(shape)),) * 2,
                matvec=lambda v, _s=shape, _L=L_envs[i], _R=R_envs[i+1], _W=W:
                    _heff_single_site(_L, _R, _W, v, _s),
                dtype=M.dtype,
            )
            vals, vecs = eigsh(op, k=max(1, min(self.n_krylov, int(np.prod(shape)) - 1)),
                               which="SA", v0=M.reshape(-1), tol=0)
            energy  = float(vals[0])
            M_new   = vecs[:, 0].reshape(shape)
            chi_l, d, chi_r = shape
            Q, R_mat = np.linalg.qr(M_new.reshape(chi_l, d * chi_r).T)
            chi_new  = Q.shape[1]
            self.mps.tensors[i].data   = Q.T.reshape(chi_new, d, chi_r)
            self.mps.tensors[i-1].data = np.tensordot(self.mps.tensors[i-1].data, R_mat.T, axes=([2], [0]))
            R_envs[i] = _update_right_env(R_envs[i+1], self.mps.tensors[i].data, W)
        return energy

    def _sweep_2s(self, L_envs, R_envs) -> float:
        L = self.mps.L
        energy = 0.0

        for i in range(L - 1):
            A, B   = self.mps.tensors[i].data, self.mps.tensors[i+1].data
            W_l, W_r = self.mpo.tensors[i].data, self.mpo.tensors[i+1].data
            chi_l, d_l, _ = A.shape;  _, d_r, chi_r = B.shape
            shape  = (chi_l, d_l, d_r, chi_r)
            dim    = chi_l * d_l * d_r * chi_r
            R_right = R_envs[i+2] if i+2 <= L else R_envs[L]
            op = LinearOperator(
                (dim, dim),
                matvec=lambda v, _s=shape, _L=L_envs[i], _R=R_right, _Wl=W_l, _Wr=W_r:
                    _heff_two_site(_L, _R, _Wl, _Wr, v, _s),
                dtype=A.dtype,
            )
            vals, vecs = eigsh(op, k=max(1, min(self.n_krylov, dim - 1)),
                               which="SA", v0=np.tensordot(A, B, axes=([2], [0])).reshape(-1), tol=0)
            energy = float(vals[0])
            theta  = vecs[:, 0].reshape(chi_l, d_l, d_r, chi_r)
            U, s, Vh = np.linalg.svd(theta.reshape(chi_l * d_l, d_r * chi_r), full_matrices=False)
            chi_new  = max(1, self.truncation.choose_bond_dim(s) if self.truncation else len(s))
            U, s, Vh = U[:, :chi_new], s[:chi_new], Vh[:chi_new, :]
            mid = Index(dim=chi_new, name=f"dmrg_bond_{i}_{i+1}", tags=frozenset({"bond"}))
            self.mps.tensors[i].data       = U.reshape(chi_l, d_l, chi_new)
            self.mps.tensors[i].indices[2] = mid
            self.mps.tensors[i+1].data       = (np.diag(s) @ Vh).reshape(chi_new, d_r, chi_r)
            self.mps.tensors[i+1].indices[0] = mid
            L_envs[i+1] = _update_left_env(L_envs[i], self.mps.tensors[i].data, W_l)

        for i in range(L - 2, -1, -1):
            A, B   = self.mps.tensors[i].data, self.mps.tensors[i+1].data
            W_l, W_r = self.mpo.tensors[i].data, self.mpo.tensors[i+1].data
            chi_l, d_l, _ = A.shape;  _, d_r, chi_r = B.shape
            shape  = (chi_l, d_l, d_r, chi_r)
            dim    = chi_l * d_l * d_r * chi_r
            R_right = R_envs[i+2] if i+2 <= L else R_envs[L]
            op = LinearOperator(
                (dim, dim),
                matvec=lambda v, _s=shape, _L=L_envs[i], _R=R_right, _Wl=W_l, _Wr=W_r:
                    _heff_two_site(_L, _R, _Wl, _Wr, v, _s),
                dtype=A.dtype,
            )
            vals, vecs = eigsh(op, k=max(1, min(self.n_krylov, dim - 1)),
                               which="SA", v0=np.tensordot(A, B, axes=([2], [0])).reshape(-1), tol=0)
            energy = float(vals[0])
            theta  = vecs[:, 0].reshape(chi_l, d_l, d_r, chi_r)
            U, s, Vh = np.linalg.svd(theta.reshape(chi_l * d_l, d_r * chi_r), full_matrices=False)
            chi_new  = max(1, self.truncation.choose_bond_dim(s) if self.truncation else len(s))
            U, s, Vh = U[:, :chi_new], s[:chi_new], Vh[:chi_new, :]
            mid = Index(dim=chi_new, name=f"dmrg_bond_{i}_{i+1}", tags=frozenset({"bond"}))
            self.mps.tensors[i].data       = (U * s[None, :]).reshape(chi_l, d_l, chi_new)
            self.mps.tensors[i].indices[2] = mid
            self.mps.tensors[i+1].data       = Vh.reshape(chi_new, d_r, chi_r)
            self.mps.tensors[i+1].indices[0] = mid
            R_envs[i+1] = _update_right_env(R_right, self.mps.tensors[i+1].data, W_r)

        self.mps.bonds = (
            [t.indices[0] for t in self.mps.tensors]
            + [self.mps.tensors[-1].indices[2]]
        )
        return energy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[float, MPS]:
        """
        Run DMRG sweeps until energy convergence or n_sweeps reached.

        Returns:
            (ground_state_energy, ground_state_MPS)
        """
        dtype  = self.mps.tensors[0].data.dtype
        L_envs = [None] * (self.mps.L + 1)
        L_envs[0] = _init_left_env(self.mps.bond_dims[0], self.mpo.bond_dims[0], dtype)
        R_envs    = _build_right_envs(self.mps, self.mpo)

        prev_energy = np.inf
        energy      = np.inf
        for _ in range(self.n_sweeps):
            energy = self._sweep(L_envs, R_envs)
            self.energies.append(energy)
            if abs(energy - prev_energy) < self.tol:
                break
            prev_energy = energy

        self.mps.normalize()
        return energy, self.mps
