"""Finite-size 2-site DMRG.

Index conventions
-----------------
MPO tensors        : (wL, d_in, d_out, wR)
MPS tensors        : (chiL, d, chiR)
Left  env  L[i]    : (chiMPS_L, chiMPS_L, chiMPO_L)  --  L[i] is the
                     boundary contraction to the *left* of site i.
Right env  R[i]    : (chiMPS_R, chiMPS_R, chiMPO_R)  --  R[i] is the
                     boundary contraction to the *right* of site i.

For the 2-site effective Hamiltonian on bond (i, i+1)::

    H_eff[(a,s1,s2,c),(b,t1,t2,d)]
        = Σ_{x,y,z} L[a,b,x] W_i[x,s1,t1,y] W_{i+1}[y,s2,t2,z] R[c,d,z]

where L = L_env[i]  (left of site i) and R = R_env[i+2] (right of site i+1).

Sweep gauge convention
----------------------
* Left-to-right half-sweep:  after optimising bond (i,i+1) the left tensor
  A_i is left-orthogonalised (U from SVD) and the singular-value weight is
  absorbed into A_{i+1} = S·Vh, making A_{i+1} the *current centre*.
  The left environment is then updated incrementally with the new A_i before
  moving to bond (i+1, i+2).

* Right-to-left half-sweep: after optimising bond (i,i+1) the right tensor
  A_{i+1} is right-orthogonalised (Vh from SVD) and the weight is absorbed
  into A_i = U·S, making A_i the current centre.  The right environment is
  updated incrementally with the new A_{i+1}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from tensor_network_library.core.canonical import _update_tensor
from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import (
    expectation_value_env,
    build_left_environments,
    build_right_environments,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class DMRGConfig:
    """Configuration for finite-size DMRG sweeps.

    Attributes
    ----------
    max_sweeps:
        Maximum number of full sweeps.
    energy_tol:
        Convergence threshold on |E_curr - E_prev| between full sweeps.
    verbose:
        Print sweep-by-sweep diagnostics.
    """

    max_sweeps: int = 20
    energy_tol: float = 1e-10
    verbose: bool = False


@dataclass
class DMRGResult:
    """Result returned by :func:`finite_dmrg`.

    Attributes
    ----------
    mps:
        Final optimised MPS.
    energies:
        Energy after each full sweep; index 0 is the initial energy.
    bond_dims:
        ``mps.bond_dims`` snapshot after each full sweep.
    """

    mps: MPS
    energies: List[float]
    bond_dims: List[List[int]]


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _build_local_heff_twosite(
    L_i: np.ndarray,
    W_i: np.ndarray,
    W_ip1: np.ndarray,
    R_ip2: np.ndarray,
) -> np.ndarray:
    """Dense 2-site effective Hamiltonian on bond (i, i+1).

    Parameters
    ----------
    L_i   : (chiL, chiL, wL)
    W_i   : (wL, d, d, wM)
    W_ip1 : (wM, d, d, wR)
    R_ip2 : (chiR, chiR, wR)

    Returns
    -------
    H_eff : matrix of shape (chiL*d*d*chiR, chiL*d*d*chiR)
    """
    H_tensor = np.einsum(
        "abx,xpqy,yrsz,cdz->aprcbqsd",
        L_i, W_i, W_ip1, R_ip2,
        optimize=True,
    )
    chiL, d1, d2, chiR = H_tensor.shape[:4]
    dim = chiL * d1 * d2 * chiR
    return H_tensor.reshape(dim, dim)


def _optimize_bond(
    L_i: np.ndarray,
    W_i: np.ndarray,
    W_ip1: np.ndarray,
    R_ip2: np.ndarray,
    A_i: np.ndarray,
    A_ip1: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Diagonalise 2-site H_eff and return the ground-state 2-site tensor."""
    chiL = A_i.shape[0]
    d    = A_i.shape[1]
    chiR = A_ip1.shape[2]

    H_eff = _build_local_heff_twosite(L_i, W_i, W_ip1, R_ip2)
    evals, evecs = np.linalg.eigh(H_eff)
    idx   = int(np.argmin(evals))
    B_opt = evecs[:, idx].reshape(chiL, d, d, chiR)
    return B_opt, float(evals[idx])


def _svd_split_left(
    B: np.ndarray, chi_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD split for the left-to-right half-sweep.

    Returns
    -------
    A_left  : left-orthogonal tensor  (chiL, d1, chi_new)   -- U
    A_right : centre tensor           (chi_new, d2, chiR)   -- S*Vh
    """
    chiL, d1, d2, chiR = B.shape
    U, S, Vh = np.linalg.svd(B.reshape(chiL * d1, d2 * chiR), full_matrices=False)
    chi_keep = min(chi_max, S.size)
    A_left  = U[:, :chi_keep].reshape(chiL, d1, chi_keep)
    A_right = (S[:chi_keep, None] * Vh[:chi_keep, :]).reshape(chi_keep, d2, chiR)
    return A_left, A_right


def _svd_split_right(
    B: np.ndarray, chi_max: int
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD split for the right-to-left half-sweep.

    Returns
    -------
    A_left  : centre tensor           (chiL, d1, chi_new)   -- U*S
    A_right : right-orthogonal tensor (chi_new, d2, chiR)   -- Vh
    """
    chiL, d1, d2, chiR = B.shape
    U, S, Vh = np.linalg.svd(B.reshape(chiL * d1, d2 * chiR), full_matrices=False)
    chi_keep = min(chi_max, S.size)
    A_left  = (U[:, :chi_keep] * S[:chi_keep]).reshape(chiL, d1, chi_keep)
    A_right = Vh[:chi_keep, :].reshape(chi_keep, d2, chiR)
    return A_left, A_right


# ---------------------------------------------------------------------------
# Incremental environment updaters
# ---------------------------------------------------------------------------


def _update_left_env(
    L_prev: np.ndarray, A: np.ndarray, W: np.ndarray
) -> np.ndarray:
    """Grow the left environment by one site.

    L_prev : (chiL, chiL, wL)
    A      : (chiL, d, chiR)   -- left-orthogonal MPS tensor
    W      : (wL, d_in, d_out, wR)

    Returns L_next : (chiR, chiR, wR)
    """
    # contract L_prev with A on chiL
    tmp  = np.einsum("abx,asa'->bxa's",  L_prev, A,       optimize=True)
    # contract with W on wL and d_in
    tmp2 = np.einsum("bxa's,xsty->bya't", tmp,   W,       optimize=True)
    # contract with conj(A) on chiL and d_out
    return  np.einsum("bya't,btb'->a'b'y", tmp2, A.conj(), optimize=True)


def _update_right_env(
    R_next: np.ndarray, A: np.ndarray, W: np.ndarray
) -> np.ndarray:
    """Shrink the right environment by one site.

    R_next : (chiR, chiR, wR)
    A      : (chiL, d, chiR)   -- right-orthogonal MPS tensor
    W      : (wL, d_in, d_out, wR)

    Returns R_prev : (chiL, chiL, wL)
    """
    # contract R_next with A on chiR
    tmp  = np.einsum("a'b'y,ata'->ayb't",  R_next, A,       optimize=True)
    # contract with W on wR and d_out
    tmp2 = np.einsum("ayb't,xsty->axb's",  tmp,    W,       optimize=True)
    # contract with conj(A) on chiR and d_in
    return  np.einsum("axb's,bsb'->abx",   tmp2,   A.conj(), optimize=True)


# ---------------------------------------------------------------------------
# Public 2-site DMRG
# ---------------------------------------------------------------------------


def finite_dmrg(
    env: Environment,
    mpo: MPO,
    mps0: MPS,
    config: DMRGConfig,
    truncation: Optional[TruncationPolicy] = None,
) -> DMRGResult:
    """Finite-size 2-site DMRG.

    Algorithm
    ---------
    1.  Build the full set of right environments from the initial MPS.
    2.  Left-to-right half-sweep over bonds (0,1), (1,2), ..., (L-2,L-1):
        * Optimise the 2-site tensor, SVD-split **left-orthogonally**.
        * Incrementally update the left environment before moving right.
    3.  Right-to-left half-sweep over bonds (L-2,L-1), ..., (0,1):
        * Optimise, SVD-split **right-orthogonally**.
        * Incrementally update the right environment before moving left.
    4.  Measure energy after the full sweep and check convergence.
    """

    env.validate_hamiltonian(mpo)
    if len(mps0) != mpo.L:
        raise ValueError(f"MPS length {len(mps0)} != MPO length {mpo.L}")
    if mps0.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"MPS phys dims {mps0.physical_dims} != MPO dims {mpo.physical_dims}"
        )

    if truncation is None:
        truncation = env.effective_truncation
    chi_max = truncation.max_bond_dim

    L = len(mps0)
    if L < 2:
        raise ValueError("finite_dmrg requires L >= 2")

    mps = mps0.copy()
    mps._assert_materialized()

    E_prev = expectation_value_env(mps, mpo)
    energies: List[float] = [E_prev]
    bond_dims: List[List[int]] = [mps.bond_dims]

    if config.verbose:
        print(f"[finite_dmrg] initial energy = {E_prev:.12f}")

    for sweep in range(1, config.max_sweeps + 1):

        # ------------------------------------------------------------------
        # Left-to-right half-sweep
        # Seed: full right environments from current MPS.
        # Left environment grown incrementally from the left boundary.
        # ------------------------------------------------------------------
        R_env   = build_right_environments(mps, mpo)  # R_env[i]: right of site i
        L_env0  = build_left_environments(mps, mpo)   # only need index 0 as seed

        L_cache = [None] * (L + 1)
        L_cache[0] = L_env0[0]

        for i in range(L - 1):
            L_i   = L_cache[i]
            R_ip2 = R_env[i + 2]
            W_i   = mpo.tensors[i].data
            W_ip1 = mpo.tensors[i + 1].data
            A_i   = mps.tensors[i].data
            A_ip1 = mps.tensors[i + 1].data

            B_opt, _       = _optimize_bond(L_i, W_i, W_ip1, R_ip2, A_i, A_ip1)
            A_i_new, A_ip1_new = _svd_split_left(B_opt, chi_max)

            _update_tensor(mps, i,     A_i_new)
            _update_tensor(mps, i + 1, A_ip1_new)

            # Grow left env with the freshly left-orthogonalised A_i_new
            L_cache[i + 1] = _update_left_env(L_i, A_i_new, W_i)

        # ------------------------------------------------------------------
        # Right-to-left half-sweep
        # Seed: full right environments from post-L2R MPS.
        # Right environment grown incrementally from the right boundary.
        # ------------------------------------------------------------------
        R_env2  = build_right_environments(mps, mpo)

        R_cache = [None] * (L + 1)
        R_cache[L] = R_env2[L]

        for i in range(L - 2, -1, -1):
            L_i   = L_cache[i]
            R_ip2 = R_cache[i + 2]
            W_i   = mpo.tensors[i].data
            W_ip1 = mpo.tensors[i + 1].data
            A_i   = mps.tensors[i].data
            A_ip1 = mps.tensors[i + 1].data

            B_opt, _       = _optimize_bond(L_i, W_i, W_ip1, R_ip2, A_i, A_ip1)
            A_i_new, A_ip1_new = _svd_split_right(B_opt, chi_max)

            _update_tensor(mps, i,     A_i_new)
            _update_tensor(mps, i + 1, A_ip1_new)

            # Grow right env with the freshly right-orthogonalised A_ip1_new
            R_cache[i + 1] = _update_right_env(R_ip2, A_ip1_new, W_ip1)

        # ------------------------------------------------------------------
        # Convergence check after the full sweep
        # ------------------------------------------------------------------
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
