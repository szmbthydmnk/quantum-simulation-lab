"""Finite-size 1-site DMRG.

Bond-dimension strategy
-----------------------
1-site DMRG cannot grow bond dimensions during the sweep because the
variational manifold is fixed by the MPS structure.  The correct
approach is to start from an MPS whose bond dimensions are already at
``chi_max`` -- random tensors work well.  Use ``MPS.from_random``.

For genuinely entangled ground states (e.g. TFIM at critical point,
Heisenberg) use 2-site DMRG instead, which SVD-truncates after each
site update and can both grow and shrink bonds.  2-site DMRG is
planned for Phase 2 of the roadmap.

Gauge maintenance
-----------------
* Initialisation  : right-canonicalise the initial MPS.
* L->R half-sweep : after optimising site i, QR -> store Q (left-
  orthonormal) at site i, absorb R into site i+1; update L_env[i+1].
* R->L half-sweep : after optimising site i, LQ -> store Q (right-
  orthonormal) at site i, absorb L into site i-1; update R_env inline.
* End of each full sweep: rebuild R_env from scratch so the next L->R
  pass always starts from a consistent set of environments.

Index convention
----------------
MPO tensors       : (wL, d_in, d_out, wR)   -- axis 1=d_in, 2=d_out
Environment arrays: (chiMPS, chiMPS, wMPO)
H_eff[(a,s,c),(b,t,d)] = sum_{x,y} L[a,b,x] * W[x,s,t,y] * R[c,d,y]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tensor_network_library.core.canonical import (
    _qr_left_step,
    _lq_right_step,
    _update_tensor,
    right_canonicalize,
)
from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.utils import (
    expectation_value_env,
    build_right_environments,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class DMRGConfig:
    """Configuration for finite-size 1-site DMRG.

    Attributes
    ----------
    max_sweeps:
        Maximum number of full sweeps (one sweep = L->R + R->L).
    energy_tol:
        Convergence threshold on |E_curr - E_prev| between full sweeps.
    verbose:
        Print sweep-by-sweep diagnostics.
    """
    max_sweeps: int   = 20
    energy_tol: float = 1e-10
    verbose:    bool  = False


@dataclass
class DMRGResult:
    """Result returned by :func:`finite_dmrg`.

    Attributes
    ----------
    mps:
        Final MPS (orthogonality centre at site 0).
    energies:
        Energy after each full sweep; index 0 is the energy before any sweep.
    bond_dims:
        ``mps.bond_dims`` snapshot after each full sweep.
    """
    mps:       MPS
    energies:  List[float]
    bond_dims: List[List[int]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_local_heff(
    L_i:   np.ndarray,
    W_i:   np.ndarray,
    R_ip1: np.ndarray,
) -> np.ndarray:
    """Dense 1-site effective Hamiltonian at site i.

    H_tensor[a,s,c, b,t,d] = L[a,b,x] * W[x,s,t,y] * R[c,d,y]

    Parameters
    ----------
    L_i   : (chiL, chiL, wL)
    W_i   : (wL, d_in, d_out, wR)
    R_ip1 : (chiR, chiR, wR)

    Returns
    -------
    Dense matrix of shape (chiL*d*chiR, chiL*d*chiR).
    """
    H_tensor = np.einsum("abx,xsty,cdy->asctbd", L_i, W_i, R_ip1, optimize=True)
    chiL = H_tensor.shape[0]
    d    = H_tensor.shape[1]
    chiR = H_tensor.shape[2]
    return H_tensor.reshape(chiL * d * chiR, chiL * d * chiR)


def _optimize_site(
    L_i:   np.ndarray,
    W_i:   np.ndarray,
    R_ip1: np.ndarray,
    A_i:   np.ndarray,
) -> tuple[np.ndarray, float]:
    """Diagonalise H_eff and return the ground-state tensor and eigenvalue.

    Parameters
    ----------
    L_i, W_i, R_ip1 : environment and MPO tensor as above.
    A_i             : current site tensor (chiL, d, chiR); shape used only
                      for the output reshape.

    Returns
    -------
    A_opt  : (chiL, d, chiR) ground-state eigenvector.
    E_local: lowest eigenvalue of H_eff.
    """
    chiL, d, chiR = A_i.shape
    H_eff = _build_local_heff(L_i, W_i, R_ip1)
    evals, evecs = np.linalg.eigh(H_eff)
    idx     = int(np.argmin(evals))
    E_local = float(evals[idx])
    A_opt   = evecs[:, idx].reshape(chiL, d, chiR)
    return A_opt, E_local


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finite_dmrg(
    env:    Environment,
    mpo:    MPO,
    mps0:   MPS,
    config: DMRGConfig,
) -> DMRGResult:
    """Finite-size 1-site DMRG.

    The bond dimensions of ``mps0`` are held fixed throughout.  To explore
    the full chi=``chi_max`` variational manifold, pass an MPS initialised
    with ``MPS.from_random(L, chi_max, ...)``.

    Parameters
    ----------
    env    : System description; ``env.validate_hamiltonian(mpo)`` is called
             at startup.
    mpo    : Hamiltonian MPO.
    mps0   : Initial MPS (right-canonicalised internally).
    config : :class:`DMRGConfig` instance.

    Returns
    -------
    DMRGResult
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    env.validate_hamiltonian(mpo)
    if len(mps0) != mpo.L:
        raise ValueError(f"MPS length {len(mps0)} != MPO length {mpo.L}")
    if mps0.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"MPS phys dims {mps0.physical_dims} != MPO dims {mpo.physical_dims}"
        )

    L = len(mps0)

    # ------------------------------------------------------------------
    # Initialise: right-canonicalise
    # ------------------------------------------------------------------
    mps   = right_canonicalize(mps0)
    R_env = build_right_environments(mps, mpo)

    # Left-environment cache: L_env_cache[i] is the left environment
    # for the bond to the left of site i, shape (chiL, chiL, wL).
    L_env_cache: list[np.ndarray] = [np.ones((1, 1, 1), dtype=mps.dtype)]

    E_prev = expectation_value_env(mps, mpo)
    energies:  list[float]     = [E_prev]
    bond_dims: list[list[int]] = [mps.bond_dims]

    if config.verbose:
        print(f"[finite_dmrg] initial energy (after right-canonicalise) = {E_prev:.12f}")

    # ------------------------------------------------------------------
    # Sweep loop
    # ------------------------------------------------------------------
    for sweep in range(1, config.max_sweeps + 1):

        # ==============================================================
        # LEFT-TO-RIGHT half-sweep  (sites 0 .. L-2)
        # ==============================================================
        for i in range(L - 1):
            A_i   = mps.tensors[i].data
            W_i   = mpo.tensors[i].data
            L_i   = L_env_cache[i]
            R_ip1 = R_env[i + 1]

            A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)

            # QR: keep Q (left-orthonormal) at site i, absorb R into i+1
            Q, R = _qr_left_step(A_opt)
            _update_tensor(mps, i, Q)
            B_new = np.tensordot(R, mps.tensors[i + 1].data, axes=([1], [0]))
            _update_tensor(mps, i + 1, B_new)

            # Grow left environment using the new left-orthonormal Q
            L_new = np.einsum(
                "abx,aic,bjd,xijy->cdy",
                L_i, Q, Q.conj(), W_i,
                optimize=True,
            )
            if len(L_env_cache) <= i + 1:
                L_env_cache.append(L_new)
            else:
                L_env_cache[i + 1] = L_new

        # Last site L->R: optimise + normalise only (no QR partner)
        i     = L - 1
        A_opt, _ = _optimize_site(
            L_env_cache[i], mpo.tensors[i].data, R_env[i + 1], mps.tensors[i].data
        )
        nrm = np.linalg.norm(A_opt.ravel())
        if nrm > 0:
            A_opt /= nrm
        _update_tensor(mps, i, A_opt)

        # ==============================================================
        # RIGHT-TO-LEFT half-sweep  (sites L-1 .. 1)
        # ==============================================================
        R_env_i = np.ones((1, 1, 1), dtype=mps.dtype)   # right boundary

        for offset in range(L - 1):
            i     = L - 1 - offset
            A_i   = mps.tensors[i].data
            W_i   = mpo.tensors[i].data
            L_i   = L_env_cache[i]
            R_ip1 = R_env_i

            A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)

            # LQ: keep Q (right-orthonormal) at site i, absorb L into i-1
            L_mat, Q = _lq_right_step(A_opt)
            _update_tensor(mps, i, Q)
            B_new = np.tensordot(mps.tensors[i - 1].data, L_mat, axes=([2], [0]))
            _update_tensor(mps, i - 1, B_new)

            # Grow right environment using the right-orthonormal Q
            R_env_i = np.einsum(
                "cdy,aic,bjd,xijy->abx",
                R_ip1, Q, Q.conj(), W_i,
                optimize=True,
            )

        # Site 0 R->L: optimise + normalise only
        A_opt, _ = _optimize_site(
            L_env_cache[0], mpo.tensors[0].data, R_env_i, mps.tensors[0].data
        )
        nrm = np.linalg.norm(A_opt.ravel())
        if nrm > 0:
            A_opt /= nrm
        _update_tensor(mps, 0, A_opt)

        # Rebuild R_env from the updated MPS for the next L->R pass.
        # This guarantees environment consistency regardless of how
        # the tensors changed during this sweep.
        R_env       = build_right_environments(mps, mpo)
        L_env_cache = [np.ones((1, 1, 1), dtype=mps.dtype)]

        # ==============================================================
        # Bookkeeping and convergence check
        # ==============================================================
        E_curr = expectation_value_env(mps, mpo)
        energies.append(E_curr)
        bond_dims.append(mps.bond_dims)

        if config.verbose:
            dE  = E_curr - E_prev
            chi = max(mps.bond_dims)
            print(
                f"[finite_dmrg] sweep {sweep:3d}: "
                f"E = {E_curr:.12f}  dE = {dE:+.3e}  chi_max = {chi}"
            )

        if abs(E_curr - E_prev) < config.energy_tol:
            if config.verbose:
                print(f"[finite_dmrg] converged after {sweep} sweeps.")
            break
        E_prev = E_curr

    return DMRGResult(mps=mps, energies=energies, bond_dims=bond_dims)
