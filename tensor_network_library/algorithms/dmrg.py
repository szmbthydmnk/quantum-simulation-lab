"""Finite-size 2-site DMRG.

This module implements the standard, robust finite-system 2-site DMRG
algorithm. It optimises pairs of neighbouring sites and performs an SVD
truncation after each update, allowing bond dimensions to grow and
shrink dynamically up to ``max_bond_dim`` from
:class:`~tensor_network_library.core.policy.TruncationPolicy`.

Index conventions
-----------------
MPO tensors       : (wL, d_in, d_out, wR)   -- axis 1=d_in, 2=d_out
MPS tensors       : (chiL, d, chiR)
Environment arrays: (chiL, chiL, wL) on the left; (chiR, chiR, wR) on the right.

For the 2-site effective Hamiltonian on bond (i,i+1)::

    H_eff[(a,s1,s2,c),(b,t1,t2,d)]
        = sum_{x,y,z} L[a,b,x] W_i[x,s1,t1,y] W_{i+1}[y,s2,t2,z] R[c,d,z]
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
        Maximum number of full sweeps (one sweep = left-to-right +
        right-to-left).
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
        Final MPS.
    energies:
        Energy after each full sweep; index 0 is the energy before any
        sweep.
    bond_dims:
        ``mps.bond_dims`` snapshot after each full sweep.
    """

    mps: MPS
    energies: List[float]
    bond_dims: List[List[int]]


# ---------------------------------------------------------------------------
# 2-site helpers
# ---------------------------------------------------------------------------


def _build_local_heff_twosite(
    L_i: np.ndarray,
    W_i: np.ndarray,
    W_ip1: np.ndarray,
    R_ip2: np.ndarray,
) -> np.ndarray:
    """Dense 2-site effective Hamiltonian on bond (i, i+1).

    Using index names::

        L_i   : L[a,b,x]
        W_i   : W_i[x,s1,t1,y]
        W_ip1 : W_{i+1}[y,s2,t2,z]
        R_ip2 : R[c,d,z]

    the tensor elements are::

        H[a,s1,s2,c,  b,t1,t2,d]
            = Σ_{x,y,z} L[a,b,x] W_i[x,s1,t1,y] W_{i+1}[y,s2,t2,z] R[c,d,z]

    The resulting tensor is reshaped to a matrix of size
    (chiL*d*d*chiR, chiL*d*d*chiR).
    """

    H_tensor = np.einsum(
        "abx,xpqy,yrsz,cdz->aprcbqsd",
        L_i,
        W_i,
        W_ip1,
        R_ip2,
        optimize=True,
    )

    chiL, d1, d2, chiR, chiL2, d1p, d2p, chiR2 = H_tensor.shape
    assert chiL2 == chiL and chiR2 == chiR and d1p == d1 and d2p == d2

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
    """Diagonalise 2-site H_eff and return the optimal 2-site tensor.

    Parameters
    ----------
    L_i, W_i, W_ip1, R_ip2 : environments and MPO tensors.
    A_i, A_ip1             : current site tensors at i and i+1
                              (shapes (chiL,d,chi_mid) and (chi_mid,d,chiR)).

    Returns
    -------
    B_opt : 4-index tensor of shape (chiL, d, d, chiR).
    E_loc : lowest eigenvalue.
    """

    chiL = A_i.shape[0]
    d = A_i.shape[1]
    chiR = A_ip1.shape[2]

    H_eff = _build_local_heff_twosite(L_i, W_i, W_ip1, R_ip2)
    evals, evecs = np.linalg.eigh(H_eff)
    idx = int(np.argmin(evals))
    E_loc = float(evals[idx])

    B_opt = evecs[:, idx].reshape(chiL, d, d, chiR)
    return B_opt, E_loc


def _svd_split(
    B: np.ndarray,
    chi_max: int,
    cutoff: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 2-site tensor into two MPS tensors via SVD.

    Args
    ----
    B:
        4-index tensor of shape (chiL, d1, d2, chiR).
    chi_max:
        Maximum allowed bond dimension after truncation.
    cutoff:
        (Reserved) spectral cutoff on discarded weight; currently unused.

    Returns
    -------
    A_i_new, A_ip1_new
        New MPS tensors of shapes (chiL, d1, chi_mid) and
        (chi_mid, d2, chiR).
    """

    chiL, d1, d2, chiR = B.shape
    M = B.reshape(chiL * d1, d2 * chiR)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    chi_keep = min(chi_max, S.size)
    U = U[:, :chi_keep]
    S = S[:chi_keep]
    Vh = Vh[:chi_keep, :]

    A_i_new = U.reshape(chiL, d1, chi_keep)
    SV = (S[:, None] * Vh).reshape(chi_keep, d2, chiR)

    return A_i_new, SV


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

    This is the main production algorithm: it follows the standard two-site
    DMRG pattern (as in ITensors and Schollwöck's review), using exact
    diagonalisation of the local 2-site effective Hamiltonian and an SVD
    truncation step after each update.

    The maximum bond dimension is taken from ``truncation.max_bond_dim``;
    when ``truncation`` is ``None``, :meth:`Environment.effective_truncation`
    is used.
    """

    # Validate basic consistency
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

    # Work on a materialised copy
    mps = mps0.copy()
    mps._assert_materialized()

    E_prev = expectation_value_env(mps, mpo)
    energies: List[float] = [E_prev]
    bond_dims: List[List[int]] = [mps.bond_dims]

    if config.verbose:
        print(f"[finite_dmrg] initial energy = {E_prev:.12f}")

    for sweep in range(1, config.max_sweeps + 1):
        # --------------------
        # Left -> Right over bonds (i,i+1)
        # --------------------
        for i in range(L - 1):
            L_env = build_left_environments(mps, mpo)
            R_env = build_right_environments(mps, mpo)

            L_i = L_env[i]
            R_ip2 = R_env[i + 2]
            W_i = mpo.tensors[i].data
            W_ip1 = mpo.tensors[i + 1].data
            A_i = mps.tensors[i].data
            A_ip1 = mps.tensors[i + 1].data

            B_opt, _ = _optimize_bond(L_i, W_i, W_ip1, R_ip2, A_i, A_ip1)
            A_i_new, A_ip1_new = _svd_split(B_opt, chi_max)

            _update_tensor(mps, i, A_i_new)
            _update_tensor(mps, i + 1, A_ip1_new)

        # --------------------
        # Right -> Left over bonds (i,i+1)
        # --------------------
        for i in range(L - 2, -1, -1):
            L_env = build_left_environments(mps, mpo)
            R_env = build_right_environments(mps, mpo)

            L_i = L_env[i]
            R_ip2 = R_env[i + 2]
            W_i = mpo.tensors[i].data
            W_ip1 = mpo.tensors[i + 1].data
            A_i = mps.tensors[i].data
            A_ip1 = mps.tensors[i + 1].data

            B_opt, _ = _optimize_bond(L_i, W_i, W_ip1, R_ip2, A_i, A_ip1)
            A_i_new, A_ip1_new = _svd_split(B_opt, chi_max)

            _update_tensor(mps, i, A_i_new)
            _update_tensor(mps, i + 1, A_ip1_new)

        # --------------------
        # Convergence check
        # --------------------
        E_curr = expectation_value_env(mps, mpo)
        energies.append(E_curr)
        bond_dims.append(mps.bond_dims)

        if config.verbose:
            dE = E_curr - E_prev
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
