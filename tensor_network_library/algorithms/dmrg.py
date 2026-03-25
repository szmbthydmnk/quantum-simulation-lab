"""Finite-size 1-site DMRG.

Design notes
------------
Correct 1-site DMRG requires the MPS to be kept in a *mixed-canonical*
form with the orthogonality centre on the site being optimised.  The
environments are then trivial on the canonical side, and the local
eigenvalue problem gives the exact optimal tensor for that site.

Gauge maintenance strategy used here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Before the first sweep the MPS is brought into right-canonical form by
  a full LQ sweep (``right_canonicalize``).  Site 0 is then the
  orthogonality centre and the right environments are products of
  right-orthonormal tensors (so R[L] = 1, and R[i] contracts cleanly).

* During the LEFT-TO-RIGHT half-sweep:
  After optimising site i, QR-decompose the new tensor::

      Q, R = qr( A_opt.reshape(chiL*d, chiR) )

  Store Q (left-orthonormal) at site i, absorb R into site i+1.  The
  left environment L[i+1] is then updated using the left-orthonormal Q,
  which keeps L[i+1] = identity up to scale.

* During the RIGHT-TO-LEFT half-sweep:
  After optimising site i, LQ-decompose the new tensor and absorb L
  into site i-1.  This keeps R[i] = identity.

Index convention (W tensors)
-----------------------------
MPO tensors have axis ordering  (wL, d_in, d_out, wR)  — matching
the ``MPO`` docstring in ``core/mpo.py``.

Environment tensors have shape  (chiMPS, chiMPS, wMPO)  and are built
by the contractions in ``core/utils.py``.

Local effective Hamiltonian
---------------------------
H_eff[(a,s,c), (b,t,d)] = sum_{x,y} L[a,b,x] * W[x,s,t,y] * R[c,d,y]

where (a,b) are ket/bra left-bond indices, (c,d) right-bond, (s,t)
physical out/in, and (x,y) MPO bond.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
    build_left_environments,
    build_right_environments,
)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class DMRGConfig:
    """Configuration parameters for finite-size 1-site DMRG.

    Attributes
    ----------
    max_sweeps:
        Maximum number of full sweeps (left-to-right + right-to-left).
    energy_tol:
        Convergence threshold on the absolute energy change between
        successive full sweeps.
    verbose:
        If ``True``, print sweep-by-sweep diagnostics.
    """
    max_sweeps: int = 20
    energy_tol: float = 1e-10
    verbose: bool = False


@dataclass
class DMRGResult:
    """Result container returned by :func:`finite_dmrg`.

    Attributes
    ----------
    mps:
        Final MPS in mixed-canonical form (orthogonality centre at
        site 0 after the last right-to-left half-sweep).
    energies:
        Energy after each full sweep (first entry = initial energy).
    bond_dims:
        ``mps.bond_dims`` snapshot after each full sweep.
    """
    mps: MPS
    energies: List[float]
    bond_dims: List[List[int]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_local_heff(
    L_i: np.ndarray,
    W_i: np.ndarray,
    R_ip1: np.ndarray,
) -> np.ndarray:
    """Dense 1-site effective Hamiltonian at site i.

    Contracts the left environment ``L_i``, the MPO tensor ``W_i``, and
    the right environment ``R_ip1`` into a Hermitian matrix that acts on
    the composite index ``(chiL, d, chiR)`` of the site tensor.

    Parameters
    ----------
    L_i:
        Left environment, shape ``(chiL, chiL, wL)``.
        Convention: ``L_i[a, b, x]`` where ``a`` (``b``) is the ket
        (bra) left-bond index and ``x`` is the MPO left-bond index.
    W_i:
        MPO tensor at site i, shape ``(wL, d_in, d_out, wR)``.
    R_ip1:
        Right environment, shape ``(chiR, chiR, wR)``.
        Convention: ``R_ip1[c, d, y]`` where ``c`` (``d``) is the ket
        (bra) right-bond index.

    Returns
    -------
    H_eff:
        Dense matrix of shape ``(chiL*d*chiR, chiL*d*chiR)``.
    """
    # H_tensor[a, s, c, b, t, d] = L[a,b,x] * W[x,s,t,y] * R[c,d,y]
    H_tensor = np.einsum("abx,xsty,cdy->asctbd", L_i, W_i, R_ip1, optimize=True)
    #                                                 ^ note: axis 1=d_in, axis 2=d_out

    chiL, d, chiR = H_tensor.shape[0], H_tensor.shape[1], H_tensor.shape[2]
    dim = chiL * d * chiR
    return H_tensor.reshape(dim, dim)


def _optimize_site(
    L_i: np.ndarray,
    W_i: np.ndarray,
    R_ip1: np.ndarray,
    A_i: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the 1-site local eigenproblem and return the optimal tensor.

    Constructs ``H_eff`` from the environments and MPO tensor, solves the
    dense Hermitian eigenproblem ``H_eff |theta> = E |theta>``, and returns
    the ground-state eigenvector reshaped to ``(chiL, d, chiR)``.

    The tensor is *not* normalised here; normalisation is handled by the
    QR/LQ gauge step that immediately follows in the sweep loop.

    Parameters
    ----------
    L_i, W_i, R_ip1:
        As in :func:`_build_local_heff`.
    A_i:
        Current site tensor, shape ``(chiL, d, chiR)``.  Only the shape
        is used to determine the output reshape.

    Returns
    -------
    A_opt:
        Optimised tensor, shape ``(chiL, d, chiR)``.
    E_local:
        Lowest eigenvalue of ``H_eff``.
    """
    chiL, d, chiR = A_i.shape
    H_eff = _build_local_heff(L_i, W_i, R_ip1)

    # Dense eigensolver — cheap for typical DMRG bond dims (chiL*d*chiR ~ 2–64).
    # Replace with scipy.sparse.linalg.eigsh + LinearOperator for large chi.
    evals, evecs = np.linalg.eigh(H_eff)
    idx = np.argmin(evals)
    E_local = float(evals[idx])
    A_opt = evecs[:, idx].reshape(chiL, d, chiR)
    return A_opt, E_local


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def finite_dmrg(
    env: Environment,
    mpo: MPO,
    mps0: MPS,
    config: DMRGConfig,
) -> DMRGResult:
    """Finite-size 1-site DMRG.

    Minimises ``<psi|H|psi>`` over the space of MPS with bond dimensions
    fixed to those of ``mps0``.

    The algorithm maintains the MPS in mixed-canonical form throughout:

    1. Initialisation: right-canonicalise ``mps0`` so every tensor is
       right-orthonormal (site 0 is the orthogonality centre).
    2. Left-to-right half-sweep (sites 0 … L-2):
       * Optimise site i by diagonalising ``H_eff``.
       * QR-decompose the result: store Q at site i (now left-orthonormal),
         absorb R into site i+1.
       * Update the left environment ``L[i+1]`` using the new Q.
    3. Right-to-left half-sweep (sites L-1 … 1):
       * Optimise site i by diagonalising ``H_eff``.
       * LQ-decompose: store Q at site i (right-orthonormal), absorb L
         into site i-1.
       * Update the right environment ``R[i]`` using the new Q.
    4. After each full sweep record the energy (via
       :func:`~tensor_network_library.core.utils.expectation_value_env`)
       and the current bond dims, then check convergence.

    Parameters
    ----------
    env:
        System description (L, d, boundary conditions).  Only the
        compatibility check ``env.validate_hamiltonian`` is used; the
        truncation policy is not applied (bond dims are fixed).
    mpo:
        Hamiltonian.  Must satisfy ``env.validate_hamiltonian(mpo)``.
    mps0:
        Initial MPS guess.  Its bond dimensions are preserved throughout.
        Pass a random or product-state MPS; the routine will canonicalise
        it internally.
    config:
        :class:`DMRGConfig` controlling the number of sweeps and the
        convergence threshold.

    Returns
    -------
    DMRGResult
        Contains the final MPS, energy per sweep, and bond-dimension
        snapshots per sweep.

    Raises
    ------
    ValueError
        If environment, MPO and MPS are incompatible.
    """

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    env.validate_hamiltonian(mpo)
    if len(mps0) != mpo.L:
        raise ValueError(f"MPS length {len(mps0)} != MPO length {mpo.L}")
    if mps0.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"MPS physical dims {mps0.physical_dims} != MPO dims {mpo.physical_dims}"
        )

    L = len(mps0)

    # ------------------------------------------------------------------
    # Initialisation: right-canonicalise so environments are trivial
    # ------------------------------------------------------------------
    # After right_canonicalize, every tensor B_i satisfies B B† = I, so
    # the right environments reduce to identity matrices and the left
    # environment at bond 0 is also the identity.
    mps = right_canonicalize(mps0)

    # Build the full set of right environments from the (now canonical) MPS.
    # R_env[i] is the right environment for sites i..L-1, shape (chiR,chiR,wR).
    R_env = build_right_environments(mps, mpo)

    # Left boundary: trivial identity (shape (1,1,1))
    L_env_cache: list[np.ndarray] = [np.ones((1, 1, 1), dtype=mps.dtype)]

    # Initial energy
    E_prev = expectation_value_env(mps, mpo)
    energies: list[float] = [E_prev]
    bond_dims: list[list[int]] = [mps.bond_dims]

    if config.verbose:
        print(f"[finite_dmrg] initial energy (after right-canonicalise) = {E_prev:.12f}")

    # ------------------------------------------------------------------
    # Sweep loop
    # ------------------------------------------------------------------
    for sweep in range(1, config.max_sweeps + 1):

        # ==============================================================
        # LEFT-TO-RIGHT half-sweep (sites 0 .. L-2)
        # ==============================================================
        # R_env is still valid from the previous right-to-left pass (or
        # the initialisation).  We carry L_env_cache[i] incrementally.

        for i in range(L - 1):
            A_i = mps.tensors[i].data
            W_i = mpo.tensors[i].data
            L_i   = L_env_cache[i]
            R_ip1 = R_env[i + 1]

            # Optimise
            A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)

            # QR: keep Q (left-orthonormal) at site i, absorb R into i+1
            Q, R = _qr_left_step(A_opt)
            _update_tensor(mps, i, Q)

            B_next = mps.tensors[i + 1].data
            B_new  = np.tensordot(R, B_next, axes=([1], [0]))
            _update_tensor(mps, i + 1, B_new)

            # Grow left environment one site to the right
            L_new = np.einsum(
                "abx,aic,bjd,xijy->cdy",
                L_i,
                Q,
                Q.conj(),
                W_i,
                optimize=True,
            )
            if len(L_env_cache) <= i + 1:
                L_env_cache.append(L_new)
            else:
                L_env_cache[i + 1] = L_new

        # Last site of left-to-right: optimise only (no QR needed here;
        # the right-to-left pass will handle gauge restoration).
        i = L - 1
        A_i   = mps.tensors[i].data
        W_i   = mpo.tensors[i].data
        L_i   = L_env_cache[i]
        R_ip1 = R_env[i + 1]          # R_env[L] = trivial (1,1,1) boundary

        A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)
        # Normalise last tensor (no QR partner to absorb the norm into)
        nrm = np.linalg.norm(A_opt.ravel())
        if nrm > 0:
            A_opt /= nrm
        _update_tensor(mps, i, A_opt)

        # ==============================================================
        # RIGHT-TO-LEFT half-sweep (sites L-1 .. 1)
        # ==============================================================
        # Rebuild right environments from the freshly updated MPS.  We
        # carry R_env_i incrementally from the right boundary.
        R_env_i = np.ones((1, 1, 1), dtype=mps.dtype)  # R_env[L]

        for offset in range(L - 1):
            i = L - 1 - offset
            A_i   = mps.tensors[i].data
            W_i   = mpo.tensors[i].data
            L_i   = L_env_cache[i]
            R_ip1 = R_env_i

            # Optimise
            A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)

            # LQ: keep Q (right-orthonormal) at site i, absorb L into i-1
            L_mat, Q = _lq_right_step(A_opt)
            _update_tensor(mps, i, Q)

            B_prev = mps.tensors[i - 1].data
            B_new  = np.tensordot(B_prev, L_mat, axes=([2], [0]))
            _update_tensor(mps, i - 1, B_new)

            # Grow right environment one site to the left
            R_env_i = np.einsum(
                "cdy,aic,bjd,xijy->abx",
                R_ip1,
                Q,
                Q.conj(),
                W_i,
                optimize=True,
            )

        # First site of right-to-left: optimise only
        i = 0
        A_i   = mps.tensors[i].data
        W_i   = mpo.tensors[i].data
        L_i   = L_env_cache[0]         # L_env[0] = trivial (1,1,1) boundary
        R_ip1 = R_env_i                # right env accumulated from the right

        A_opt, _ = _optimize_site(L_i, W_i, R_ip1, A_i)
        nrm = np.linalg.norm(A_opt.ravel())
        if nrm > 0:
            A_opt /= nrm
        _update_tensor(mps, i, A_opt)

        # Refresh R_env for the next left-to-right pass
        R_env = build_right_environments(mps, mpo)
        # Trim L_env_cache to just the left boundary for the next L->R pass
        L_env_cache = [np.ones((1, 1, 1), dtype=mps.dtype)]

        # ==============================================================
        # End-of-sweep bookkeeping
        # ==============================================================
        E_curr = expectation_value_env(mps, mpo)
        energies.append(E_curr)
        bond_dims.append(mps.bond_dims)

        if config.verbose:
            dE = E_curr - E_prev
            print(f"[finite_dmrg] sweep {sweep:3d}: E = {E_curr:.12f}  dE = {dE:+.3e}")

        if abs(E_curr - E_prev) < config.energy_tol:
            if config.verbose:
                print(f"[finite_dmrg] converged after {sweep} sweeps.")
            break
        E_prev = E_curr

    return DMRGResult(mps=mps, energies=energies, bond_dims=bond_dims)
