"""Finite-size 1-site DMRG with subspace expansion.

Design notes
------------
Correct 1-site DMRG requires the MPS to be kept in a *mixed-canonical*
form with the orthogonality centre on the site being optimised.  The
environments are then trivial on the canonical side, and the local
eigenvalue problem gives the exact optimal tensor for that site.

Bond-dimension growth (subspace expansion)
------------------------------------------
Pure 1-site DMRG cannot grow bond dimensions: the variational manifold
is fixed by the initial chi.  We use *perturbative subspace expansion*
(Hubig et al., Phys. Rev. B 91, 155115, 2015): before the QR/LQ gauge
step, the optimised tensor is padded with a small random block so that
the bond can grow by up to ``expand_step`` columns/rows per sweep, up
to the ``chi_max`` cap.  The perturbation amplitude decays geometrically
as ``noise * noise_decay^sweep`` so it does not destabilise a converged
solution.

Gauge maintenance
~~~~~~~~~~~~~~~~~
* Before the first sweep: right-canonicalise ``mps0``.
* L->R half-sweep: QR after each site; absorb R into site i+1;
  update L_env[i+1] from the left-orthonormal Q.
* R->L half-sweep: LQ after each site; absorb L into site i-1;
  update R_env inline.

Index convention (W tensors)
-----------------------------
MPO tensors:        (wL, d_in, d_out, wR)   -- axis 1=d_in, axis 2=d_out
Environment tensors: (chiMPS, chiMPS, wMPO)

H_eff[(a,s,c),(b,t,d)] = sum_{x,y} L[a,b,x] * W[x,s,t,y] * R[c,d,y]
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    """Configuration for finite-size 1-site DMRG with subspace expansion.

    Attributes
    ----------
    max_sweeps:
        Maximum number of full sweeps.
    energy_tol:
        Convergence threshold on |E_curr - E_prev| between full sweeps.
    chi_max:
        Maximum bond dimension.  ``None`` means no cap (use with care).
    noise:
        Initial amplitude of the random perturbation added during subspace
        expansion.  Set to 0.0 to disable expansion (fixed bond dims).
    noise_decay:
        Multiplicative decay applied to ``noise`` after each full sweep.
        0.9 is a good default: the perturbation shrinks as DMRG converges.
    expand_step:
        Maximum number of new singular vectors added per bond per sweep.
    verbose:
        Print sweep-by-sweep diagnostics.
    seed:
        RNG seed for the perturbation noise.  ``None`` = non-deterministic.
    """
    max_sweeps:   int   = 20
    energy_tol:   float = 1e-10
    chi_max:      Optional[int] = None
    noise:        float = 1e-3
    noise_decay:  float = 0.9
    expand_step:  int   = 2
    verbose:      bool  = False
    seed:         Optional[int] = None


@dataclass
class DMRGResult:
    """Result returned by :func:`finite_dmrg`.

    Attributes
    ----------
    mps:
        Final MPS (orthogonality centre at site 0 after the last R->L pass).
    energies:
        Energy after each full sweep (index 0 = before any sweep).
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
    """Dense 1-site effective Hamiltonian.

    H_tensor[a,s,c, b,t,d] = L[a,b,x] * W[x,s,t,y] * R[c,d,y]

    Args:
        L_i:   (chiL, chiL, wL)
        W_i:   (wL, d_in, d_out, wR)
        R_ip1: (chiR, chiR, wR)

    Returns:
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
    """Solve the 1-site local eigenproblem.

    Returns the ground-state eigenvector reshaped to (chiL, d, chiR) and
    the corresponding eigenvalue.
    """
    chiL, d, chiR = A_i.shape
    H_eff = _build_local_heff(L_i, W_i, R_ip1)
    evals, evecs = np.linalg.eigh(H_eff)
    idx     = int(np.argmin(evals))
    E_local = float(evals[idx])
    A_opt   = evecs[:, idx].reshape(chiL, d, chiR)
    return A_opt, E_local


def _expand_right(
    A:          np.ndarray,
    B_next:     np.ndarray,
    chi_max:    int,
    noise:      float,
    expand_step: int,
    rng:        np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subspace expansion on the right bond of site i during L->R sweep.

    Pads A with ``k`` extra random columns (k <= expand_step) so that the
    right bond can grow from chiR to min(chiR + k, chi_max).  The companion
    tensor B_next is padded with k zero rows so the contraction remains
    valid.

    Args:
        A:          Current optimised tensor, shape (chiL, d, chiR).
        B_next:     Next-site tensor, shape (chiR, d', chiR').
        chi_max:    Hard cap on the new bond dimension.
        noise:      Amplitude of random padding columns.
        expand_step: Max columns to add in this call.
        rng:        Numpy random generator.

    Returns:
        A_exp:      Padded tensor, shape (chiL, d, chiR_new).
        B_exp:      Padded next tensor, shape (chiR_new, d', chiR').
    """
    chiL, d, chiR = A.shape
    chiR_next, d_next, chiR_right = B_next.shape
    assert chiR == chiR_next, "bond mismatch"

    k = min(expand_step, chi_max - chiR)
    if k <= 0:
        return A, B_next

    # Random noise block: shape (chiL*d, k), then reshape to (chiL, d, k)
    noise_block = noise * rng.standard_normal((chiL * d, k)).astype(A.dtype)
    if np.issubdtype(A.dtype, np.complexfloating):
        noise_block = noise_block + 1j * noise * rng.standard_normal((chiL * d, k)).astype(A.dtype)
    noise_block = noise_block.reshape(chiL, d, k)

    A_exp = np.concatenate([A, noise_block], axis=2)          # (chiL, d, chiR+k)
    B_pad = np.zeros((k, d_next, chiR_right), dtype=B_next.dtype)
    B_exp = np.concatenate([B_next, B_pad], axis=0)            # (chiR+k, d', chiR')
    return A_exp, B_exp


def _expand_left(
    A:          np.ndarray,
    B_prev:     np.ndarray,
    chi_max:    int,
    noise:      float,
    expand_step: int,
    rng:        np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Subspace expansion on the left bond of site i during R->L sweep.

    Pads A with ``k`` extra random rows on the left bond and pads B_prev
    with k zero columns.

    Args:
        A:          Current optimised tensor, shape (chiL, d, chiR).
        B_prev:     Previous-site tensor, shape (chiL', d', chiL).
        chi_max:    Hard cap on the new bond dimension.
        noise:      Amplitude of random padding rows.
        expand_step: Max rows to add.
        rng:        Numpy random generator.

    Returns:
        A_exp:      Padded tensor, shape (chiL_new, d, chiR).
        B_exp:      Padded prev tensor, shape (chiL', d', chiL_new).
    """
    chiL, d, chiR = A.shape
    chiL_left, d_prev, chiL_right = B_prev.shape
    assert chiL == chiL_right, "bond mismatch"

    k = min(expand_step, chi_max - chiL)
    if k <= 0:
        return A, B_prev

    noise_block = noise * rng.standard_normal((k, d * chiR)).astype(A.dtype)
    if np.issubdtype(A.dtype, np.complexfloating):
        noise_block = noise_block + 1j * noise * rng.standard_normal((k, d * chiR)).astype(A.dtype)
    noise_block = noise_block.reshape(k, d, chiR)

    A_exp = np.concatenate([A, noise_block], axis=0)           # (chiL+k, d, chiR)
    B_pad = np.zeros((chiL_left, d_prev, k), dtype=B_prev.dtype)
    B_exp = np.concatenate([B_prev, B_pad], axis=2)            # (chiL', d', chiL+k)
    return A_exp, B_exp


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finite_dmrg(
    env:    Environment,
    mpo:    MPO,
    mps0:   MPS,
    config: DMRGConfig,
) -> DMRGResult:
    """Finite-size 1-site DMRG with perturbative subspace expansion.

    Parameters
    ----------
    env:
        System description.  ``env.validate_hamiltonian(mpo)`` is called
        at the start.  ``env.max_bond_dim`` is used as the default
        ``chi_max`` when ``config.chi_max`` is None.
    mpo:
        Hamiltonian MPO.
    mps0:
        Initial MPS guess (will be right-canonicalised internally).
    config:
        :class:`DMRGConfig` instance.

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

    L       = len(mps0)
    chi_max = config.chi_max if config.chi_max is not None else env.max_bond_dim
    rng     = np.random.default_rng(config.seed)
    noise   = config.noise

    # ------------------------------------------------------------------
    # Initialise: right-canonicalise
    # ------------------------------------------------------------------
    mps   = right_canonicalize(mps0)
    R_env = build_right_environments(mps, mpo)
    L_env_cache: list[np.ndarray] = [np.ones((1, 1, 1), dtype=mps.dtype)]

    E_prev = expectation_value_env(mps, mpo)
    energies:  list[float]      = [E_prev]
    bond_dims: list[list[int]]  = [mps.bond_dims]

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

            # Subspace expansion: pad right bond before QR
            B_next = mps.tensors[i + 1].data
            A_opt, B_next_exp = _expand_right(
                A_opt, B_next, chi_max, noise, config.expand_step, rng
            )

            # QR: left-orthonormal Q stays at site i, R absorbed into i+1
            Q, R = _qr_left_step(A_opt)
            _update_tensor(mps, i, Q)
            B_new = np.tensordot(R, B_next_exp, axes=([1], [0]))
            _update_tensor(mps, i + 1, B_new)

            # Grow left environment using the left-orthonormal Q
            L_new = np.einsum(
                "abx,aic,bjd,xijy->cdy",
                L_i, Q, Q.conj(), W_i,
                optimize=True,
            )
            if len(L_env_cache) <= i + 1:
                L_env_cache.append(L_new)
            else:
                L_env_cache[i + 1] = L_new

        # Last site L->R: optimise only, normalise, no expansion
        i     = L - 1
        A_i   = mps.tensors[i].data
        W_i   = mpo.tensors[i].data
        L_i   = L_env_cache[i]
        R_ip1 = R_env[i + 1]          # trivial (1,1,1) right boundary

        A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)
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

            # Subspace expansion: pad left bond before LQ
            B_prev = mps.tensors[i - 1].data
            A_opt, B_prev_exp = _expand_left(
                A_opt, B_prev, chi_max, noise, config.expand_step, rng
            )

            # LQ: right-orthonormal Q stays at site i, L absorbed into i-1
            L_mat, Q = _lq_right_step(A_opt)
            _update_tensor(mps, i, Q)
            B_new = np.tensordot(B_prev_exp, L_mat, axes=([2], [0]))
            _update_tensor(mps, i - 1, B_new)

            # Grow right environment using the right-orthonormal Q
            R_env_i = np.einsum(
                "cdy,aic,bjd,xijy->abx",
                R_ip1, Q, Q.conj(), W_i,
                optimize=True,
            )

        # Site 0 R->L: optimise only, normalise
        i     = 0
        A_i   = mps.tensors[i].data
        W_i   = mpo.tensors[i].data
        L_i   = L_env_cache[0]       # trivial (1,1,1) left boundary
        R_ip1 = R_env_i

        A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)
        nrm = np.linalg.norm(A_opt.ravel())
        if nrm > 0:
            A_opt /= nrm
        _update_tensor(mps, i, A_opt)

        # Refresh environments for next L->R pass
        R_env       = build_right_environments(mps, mpo)
        L_env_cache = [np.ones((1, 1, 1), dtype=mps.dtype)]

        # Decay noise for next sweep
        noise *= config.noise_decay

        # ==============================================================
        # Bookkeeping and convergence
        # ==============================================================
        E_curr = expectation_value_env(mps, mpo)
        energies.append(E_curr)
        bond_dims.append(mps.bond_dims)

        if config.verbose:
            dE   = E_curr - E_prev
            chi  = max(mps.bond_dims)
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
