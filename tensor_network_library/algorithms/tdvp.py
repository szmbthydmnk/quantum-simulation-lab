"""
Finite-size 1-site and 2-site TDVP for MPS.

Index conventions
-----------------
MPO tensors        : (wL, d_in, d_out, wR)
MPS tensors        : (chiL, d, chiR)
Left  env  L[i]    : (chiMPS, chiMPS, chiMPO)  -- contraction to the *left* of site i
Right env  R[i]    : (chiMPS, chiMPS, chiMPO)  -- contraction to the *right* of site i

For the 1-site effective Hamiltonian at site i::

    H1_eff[(a,s,c),(b,t,d)] = sum_{x,y} L[a,b,x] W_i[x,s,t,y] R[c,d,y]

For the zero-site (bond) effective Hamiltonian on bond (i, i+1)::

    H0_eff[(a,c),(b,d)] = sum_{x} L[a,b,x] R[c,d,x]

For the 2-site effective Hamiltonian on bond (i, i+1) (same as DMRG)::

    H2_eff[(a,s1,s2,c),(b,t1,t2,d)]
        = sum_{x,y,z} L[a,b,x] W_i[x,s1,t1,y] W_{i+1}[y,s2,t2,z] R[c,d,z]

Sweep gauge convention
----------------------
* Both algorithms begin from a right-canonical MPS with the orthogonality
  centre at site 0.  The orthogonality centre migrates rightward during the
  right half-sweep and leftward during the left half-sweep.
* 1TDVP right half-sweep at site i:
  (a) Evolve centre M_i forward by dt/2 with H1_eff  (M_i -> M_i').
  (b) QR-factorise: M_i' = Q R.  Set site tensor <- Q.
  (c) Evolve bond centre C = R backward by dt/2 with H0_eff.
  (d) Absorb C into site i+1 to obtain the new orthogonality centre.
* 1TDVP left half-sweep: analogous with LQ factorisation and backward
  bond evolution using the right bond centre.
* 2TDVP right half-sweep at bond (i, i+1):
  (a) Form two-site tensor T = M_i . M_{i+1}.
  (b) Evolve T forward by dt/2 with H2_eff.
  (c) SVD-split T (with truncation); absorb left factor into site i.
  (d) Evolve the right singular-value / Vh combination backward by dt/2
      with H1_eff at site i+1 (bond backward step in 2TDVP).
* Full step for both variants uses a symmetric Strang splitting:
  right half-sweep (dt/2) + left half-sweep (dt/2).

Krylov exponentiation
---------------------
The action of exp(A v) on a vector v is approximated by a Lanczos/Arnoldi
procedure of dimension ``krylov_dim``.  For Hermitian H_eff the Lanczos
recurrence is used; the resulting small Hessenberg matrix is exponentiated
exactly with ``scipy.linalg.expm``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg

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
class TDVPConfig:
    """Configuration for finite-size TDVP time evolution.

    Attributes
    ----------
    dt:
        Complex or real time step.  Use ``dt = -1j * tau`` for imaginary-time
        evolution; use ``dt`` real for real-time evolution.
    num_steps:
        Number of time steps to perform.
    krylov_dim:
        Size of the Krylov subspace used to approximate the local matrix
        exponential action.  Values in the range 10--30 are typically
        sufficient.  Increasing this improves accuracy of the local solve
        at the cost of additional matrix-vector products.
    verbose:
        Print per-step diagnostics.
    """

    dt: complex = 0.01
    num_steps: int = 100
    krylov_dim: int = 20
    verbose: bool = False


@dataclass
class TDVPResult:
    """Result returned by :func:`tdvp1` or :func:`tdvp2`.

    Attributes
    ----------
    mps:
        MPS at the final time.
    times:
        1-D array of shape ``(num_steps + 1,)`` containing the time at
        each stored snapshot (including t = 0).
    energies:
        Expectation value of the Hamiltonian at each stored time.
    norms:
        MPS norm at each stored time.  For 1TDVP this should remain
        machine-precision close to 1 throughout.
    bond_dims:
        ``mps.bond_dims`` snapshot at each stored time.
    """

    mps: MPS
    times: np.ndarray
    energies: List[float] = field(default_factory=list)
    norms: List[float] = field(default_factory=list)
    bond_dims: List[List[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Krylov matrix-exponential action:  exp(dt * H) |v>
# ---------------------------------------------------------------------------


def _expm_krylov(
    matvec,
    v: np.ndarray,
    dt: complex,
    krylov_dim: int,
) -> np.ndarray:
    """Approximate exp(dt * H) |v> via a Lanczos/Krylov projection.

    The effective Hamiltonians arising in TDVP are Hermitian, so the
    Lanczos three-term recurrence is used to build an orthonormal basis
    {q_0, ..., q_{m-1}} for the Krylov subspace K_m(H, v).  The small
    tri-diagonal matrix T_m is exponentiated exactly, and the result is
    projected back to the original space.

    Parameters
    ----------
    matvec:
        Callable with signature ``w = matvec(v)`` implementing the action
        of H on a flattened real or complex vector v.
    v:
        Starting vector; will be normalised internally.
    dt:
        Exponent prefactor (imaginary for real-time, real for imaginary-time).
    krylov_dim:
        Maximum number of Lanczos steps.

    Returns
    -------
    Approximation of exp(dt * H) |v>, same shape as v.
    """
    n = v.size
    dim = min(krylov_dim, n)

    norm_v = np.linalg.norm(v)
    if norm_v == 0.0:
        return v.copy()

    Q = np.zeros((n, dim), dtype=complex)
    alpha = np.zeros(dim, dtype=complex)
    beta  = np.zeros(dim - 1, dtype=complex)

    Q[:, 0] = v.ravel() / norm_v

    for j in range(dim):
        w = matvec(Q[:, j]).ravel().astype(complex)
        alpha[j] = np.dot(Q[:, j].conj(), w)
        w -= alpha[j] * Q[:, j]
        if j > 0:
            w -= beta[j - 1] * Q[:, j - 1]

        if j < dim - 1:
            b = np.linalg.norm(w)
            beta[j] = b
            if b < 1e-14:          # Krylov space exhausted
                dim = j + 1
                Q   = Q[:, :dim]
                alpha = alpha[:dim]
                beta  = beta[:j]
                break
            Q[:, j + 1] = w / b

    # Build and exponentiate the tridiagonal Hessenberg matrix
    T = np.diag(alpha[:dim]) + np.diag(beta[:dim - 1], 1) + np.diag(beta[:dim - 1].conj(), -1)
    e1 = np.zeros(dim, dtype=complex)
    e1[0] = 1.0
    expT_e1 = scipy.linalg.expm(dt * T) @ e1

    return (norm_v * Q[:, :dim] @ expT_e1).reshape(v.shape)


# ---------------------------------------------------------------------------
# Effective Hamiltonian builders
# ---------------------------------------------------------------------------


def _build_heff_onesite(
    L_i: np.ndarray,
    W_i: np.ndarray,
    R_i: np.ndarray,
) -> np.ndarray:
    """Dense 1-site effective Hamiltonian at site i.

    Parameters
    ----------
    L_i : (chiL, chiL, wL)
    W_i : (wL, d, d, wR)
    R_i : (chiR, chiR, wR)   -- R to the *right* of site i

    Returns
    -------
    H1_eff : matrix of shape (chiL*d*chiR, chiL*d*chiR)
    """
    # H1_tensor[a,s,c,b,t,d] = sum_{x,y} L[a,b,x] W[x,s,t,y] R[c,d,y]
    H1 = np.einsum("abx,xsty,cdy->asctbd", L_i, W_i, R_i, optimize=True)
    chiL, d = H1.shape[0], H1.shape[1]
    chiR    = H1.shape[2]
    dim     = chiL * d * chiR
    return H1.reshape(dim, dim)


def _build_heff_zerosite(
    L_i: np.ndarray,
    R_ip1: np.ndarray,
) -> np.ndarray:
    """Dense zero-site (bond) effective Hamiltonian on bond (i, i+1).

    This operator acts on the bond-centre matrix C of shape (chiL, chiR).

    Parameters
    ----------
    L_i   : (chiL, chiL, wL)   -- left env to the *left* of site i+1
    R_ip1 : (chiR, chiR, wR)   -- right env to the *right* of site i

    Returns
    -------
    H0_eff : matrix of shape (chiL*chiR, chiL*chiR)
    """
    # H0[a,c,b,d] = sum_x L[a,b,x] R[c,d,x]
    H0 = np.einsum("abx,cdx->acbd", L_i, R_ip1, optimize=True)
    chiL = H0.shape[0]
    chiR = H0.shape[1]
    dim  = chiL * chiR
    return H0.reshape(dim, dim)


def _build_heff_twosite(
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
    H2_eff : matrix of shape (chiL*d*d*chiR, chiL*d*d*chiR)
    """
    H2 = np.einsum(
        "abx,xpqy,yrsz,cdz->aprctbsd",
        L_i, W_i, W_ip1, R_ip2,
        optimize=True,
    )
    chiL, d1, d2, chiR = H2.shape[0], H2.shape[1], H2.shape[2], H2.shape[3]
    dim = chiL * d1 * d2 * chiR
    return H2.reshape(dim, dim)


# ---------------------------------------------------------------------------
# Incremental environment updaters  (identical conventions to dmrg.py)
# ---------------------------------------------------------------------------


def _update_left_env(
    L_prev: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Grow the left environment by one left-orthogonal site tensor.

    Parameters
    ----------
    L_prev : (chiL, chiL, wL)
    A      : (chiL, d, chiR)  -- left-orthogonal MPS tensor at site i
    W      : (wL, d, d, wR)   -- MPO tensor at site i

    Returns
    -------
    L_next : (chiR, chiR, wR)
    """
    tmp  = np.einsum("abx,asc->bxsc",   L_prev, A,       optimize=True)
    tmp2 = np.einsum("bxsc,xsty->btyc", tmp,    W,       optimize=True)
    return  np.einsum("btyc,bte->cey",   tmp2,   A.conj(), optimize=True)


def _update_right_env(
    R_next: np.ndarray,
    A: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Shrink the right environment by one right-orthogonal site tensor.

    Parameters
    ----------
    R_next : (chiR, chiR, wR)
    A      : (chiL, d, chiR)  -- right-orthogonal MPS tensor at site i
    W      : (wL, d, d, wR)   -- MPO tensor at site i

    Returns
    -------
    R_prev : (chiL, chiL, wL)
    """
    tmp  = np.einsum("cey,asc->asey",   R_next, A,       optimize=True)
    tmp2 = np.einsum("asey,xsty->axte", tmp,    W,       optimize=True)
    return  np.einsum("axte,bte->abx",   tmp2,   A.conj(), optimize=True)


# ---------------------------------------------------------------------------
# Helper: MPS tensor accessor
# ---------------------------------------------------------------------------


def _get(mps: MPS, i: int) -> np.ndarray:
    return mps.tensors[i].data


def _set(mps: MPS, i: int, data: np.ndarray) -> None:
    mps.tensors[i].data = data


# ---------------------------------------------------------------------------
# 1TDVP  --  single-site update with backward bond evolution
# ---------------------------------------------------------------------------


def _onesite_forward(
    M: np.ndarray,
    L_i: np.ndarray,
    W_i: np.ndarray,
    R_i: np.ndarray,
    dt: complex,
    krylov_dim: int,
) -> np.ndarray:
    """Evolve the one-site centre tensor M forward by dt.

    Uses a Krylov approximation to exp(- i * H1_eff * dt) |M>.
    For imaginary-time evolution pass dt = -tau (real and negative).
    The sign convention follows -i H t for real time, and the caller is
    responsible for incorporating the factor of -i into dt.

    Parameters
    ----------
    M   : (chiL, d, chiR)  -- current centre tensor
    L_i : left env to the left of site i
    W_i : MPO tensor at site i
    R_i : right env to the right of site i
    dt  : time step (complex)

    Returns
    -------
    M_evolved : (chiL, d, chiR)
    """
    chiL, d, chiR = M.shape
    H1 = _build_heff_onesite(L_i, W_i, R_i)

    def matvec(v):
        return H1 @ v

    M_new = _expm_krylov(matvec, M.astype(complex), -1j * dt, krylov_dim)
    return M_new.reshape(chiL, d, chiR)


def _zerosite_backward(
    C: np.ndarray,
    L_ip1: np.ndarray,
    R_ip1: np.ndarray,
    dt: complex,
    krylov_dim: int,
) -> np.ndarray:
    """Evolve the zero-site bond centre C *backward* by dt.

    The backward sign arises from the tangent-space projector decomposition:
    the bond-centre term in the projector carries an opposite sign relative
    to the site-centre term, yielding evolution under +i H0_eff dt instead
    of -i H0_eff dt.

    Parameters
    ----------
    C     : (chiL, chiR)  -- bond centre matrix
    L_ip1 : left env to the *left* of site i+1  (i.e. directly to the right of site i)
    R_ip1 : right env to the *right* of site i  (i.e. directly to the left of site i+1)
    dt    : time step

    Returns
    -------
    C_evolved : (chiL, chiR)
    """
    chiL, chiR = C.shape
    H0 = _build_heff_zerosite(L_ip1, R_ip1)

    def matvec(v):
        return H0 @ v

    C_new = _expm_krylov(matvec, C.astype(complex), +1j * dt, krylov_dim)
    return C_new.reshape(chiL, chiR)


def _onesite_sweep_right(
    mps: MPS,
    mpo: MPO,
    L_cache: list,
    R_env: list,
    dt: complex,
    krylov_dim: int,
) -> None:
    """Right half-sweep for 1TDVP: site 0 -> site L-1.

    Sweeps the orthogonality centre from site 0 to site L-1 while
    performing the forward one-site and backward bond updates at each
    intermediate bond.

    Parameters
    ----------
    mps        : MPS in right-canonical form with centre at site 0.
    mpo        : MPO Hamiltonian.
    L_cache    : list of length L+1; L_cache[0] must be the left boundary.
                 Entries are filled in-place during the sweep.
    R_env      : pre-built right environments; R_env[i] is to the right of site i.
    dt         : half-step for this sweep.
    krylov_dim : Krylov subspace dimension.
    """
    L = len(mps)
    for i in range(L - 1):
        M = _get(mps, i)
        L_i = L_cache[i]
        W_i = mpo.tensors[i].data
        R_i = R_env[i + 1]

        # Forward 1-site evolution
        M_new = _onesite_forward(M, L_i, W_i, R_i, dt, krylov_dim)

        # QR factorisation: orthogonality centre passes to i+1
        chiL, d, chiR = M_new.shape
        Q, R = np.linalg.qr(M_new.reshape(chiL * d, chiR))
        chi_new = Q.shape[1]
        _set(mps, i, Q.reshape(chiL, d, chi_new))

        # Update left environment with the freshly left-orthogonalised tensor
        L_cache[i + 1] = _update_left_env(L_i, Q.reshape(chiL, d, chi_new), W_i)

        # Backward zero-site evolution of the bond centre C = R
        C = R  # shape (chi_new, chiR)
        L_ip1 = L_cache[i + 1]
        R_ip1 = R_env[i + 1]   # right env to the right of site i  (= right of bond)
        C_new = _zerosite_backward(C, L_ip1, R_ip1, dt, krylov_dim)

        # Absorb evolved bond centre into the right site
        M_next = _get(mps, i + 1)   # shape (chiR, d_next, chiR_next)
        _set(mps, i + 1, np.tensordot(C_new, M_next, axes=([1], [0])))

    # Evolve rightmost site tensor (no backward bond step at boundary)
    i = L - 1
    M = _get(mps, i)
    M_new = _onesite_forward(M, L_cache[i], mpo.tensors[i].data, R_env[i + 1], dt, krylov_dim)
    _set(mps, i, M_new)


def _onesite_sweep_left(
    mps: MPS,
    mpo: MPO,
    L_cache: list,
    R_cache: list,
    dt: complex,
    krylov_dim: int,
) -> None:
    """Left half-sweep for 1TDVP: site L-1 -> site 0.

    Parameters
    ----------
    mps        : MPS with centre at site L-1 after the right half-sweep.
    mpo        : MPO Hamiltonian.
    L_cache    : left environments filled during the right sweep.
    R_cache    : list of length L+1; R_cache[L] must be the right boundary.
                 Entries are filled in-place during the sweep.
    dt         : half-step for this sweep.
    krylov_dim : Krylov subspace dimension.
    """
    L = len(mps)
    for i in range(L - 1, 0, -1):
        M = _get(mps, i)
        L_i = L_cache[i]
        W_i = mpo.tensors[i].data
        R_i = R_cache[i + 1]

        # Forward 1-site evolution
        M_new = _onesite_forward(M, L_i, W_i, R_i, dt, krylov_dim)

        # LQ factorisation: orthogonality centre passes to i-1
        chiL, d, chiR = M_new.shape
        L_mat, Q = scipy.linalg.lq(M_new.reshape(chiL, d * chiR))
        chi_new = Q.shape[0]
        _set(mps, i, Q.reshape(chi_new, d, chiR))

        # Update right environment with the freshly right-orthogonalised tensor
        R_cache[i] = _update_right_env(R_i, Q.reshape(chi_new, d, chiR), W_i)

        # Backward zero-site evolution of the bond centre C = L_mat
        C = L_mat   # shape (chiL, chi_new)
        L_im1 = L_cache[i]
        R_im1 = R_cache[i]
        C_new = _zerosite_backward(C, L_im1, R_im1, dt, krylov_dim)

        # Absorb evolved bond centre into the left site
        M_prev = _get(mps, i - 1)   # shape (chiL_prev, d_prev, chiL)
        _set(mps, i - 1, np.tensordot(M_prev, C_new, axes=([2], [0])))

    # Evolve leftmost site tensor
    i = 0
    M = _get(mps, i)
    M_new = _onesite_forward(M, L_cache[0], mpo.tensors[0].data, R_cache[1], dt, krylov_dim)
    _set(mps, i, M_new)


# ---------------------------------------------------------------------------
# 2TDVP  --  two-site update with truncated SVD for bond-dimension control
# ---------------------------------------------------------------------------


def _twosite_forward(
    T: np.ndarray,
    L_i: np.ndarray,
    W_i: np.ndarray,
    W_ip1: np.ndarray,
    R_ip2: np.ndarray,
    dt: complex,
    krylov_dim: int,
) -> np.ndarray:
    """Evolve the two-site tensor T forward by dt with H2_eff.

    Parameters
    ----------
    T     : (chiL, d1, d2, chiR)  -- merged two-site tensor
    L_i   : left env to the left of site i
    W_i   : MPO tensor at site i
    W_ip1 : MPO tensor at site i+1
    R_ip2 : right env to the right of site i+1
    dt    : time step

    Returns
    -------
    T_evolved : (chiL, d1, d2, chiR)
    """
    chiL, d1, d2, chiR = T.shape
    H2 = _build_heff_twosite(L_i, W_i, W_ip1, R_ip2)

    def matvec(v):
        return H2 @ v

    T_new = _expm_krylov(matvec, T.astype(complex), -1j * dt, krylov_dim)
    return T_new.reshape(chiL, d1, d2, chiR)


def _svd_split_left_trunc(
    T: np.ndarray,
    chi_max: int,
    svd_tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD split with truncation for the 2TDVP right half-sweep.

    Returns
    -------
    A_left  : left-orthogonal tensor  (chiL, d1, chi_new)  -- U
    SC      : centre + right factor   (chi_new, d2, chiR)  -- S * Vh, non-orthogonal
    """
    chiL, d1, d2, chiR = T.shape
    U, S, Vh = np.linalg.svd(T.reshape(chiL * d1, d2 * chiR), full_matrices=False)
    if svd_tol > 0.0:
        keep = np.sum(S > svd_tol * S[0])
        chi_new = max(1, min(chi_max, int(keep)))
    else:
        chi_new = min(chi_max, S.size)
    A_left = U[:, :chi_new].reshape(chiL, d1, chi_new)
    SC     = (S[:chi_new, None] * Vh[:chi_new, :]).reshape(chi_new, d2, chiR)
    return A_left, SC


def _svd_split_right_trunc(
    T: np.ndarray,
    chi_max: int,
    svd_tol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """SVD split with truncation for the 2TDVP left half-sweep.

    Returns
    -------
    SC      : centre + left factor    (chiL, d1, chi_new)  -- U * S, non-orthogonal
    A_right : right-orthogonal tensor (chi_new, d2, chiR)  -- Vh
    """
    chiL, d1, d2, chiR = T.shape
    U, S, Vh = np.linalg.svd(T.reshape(chiL * d1, d2 * chiR), full_matrices=False)
    if svd_tol > 0.0:
        keep = np.sum(S > svd_tol * S[0])
        chi_new = max(1, min(chi_max, int(keep)))
    else:
        chi_new = min(chi_max, S.size)
    SC      = (U[:, :chi_new] * S[:chi_new]).reshape(chiL, d1, chi_new)
    A_right = Vh[:chi_new, :].reshape(chi_new, d2, chiR)
    return SC, A_right


def _twosite_sweep_right(
    mps: MPS,
    mpo: MPO,
    L_cache: list,
    R_env: list,
    dt: complex,
    krylov_dim: int,
    chi_max: int,
    svd_tol: float,
) -> None:
    """Right half-sweep for 2TDVP: bond (0,1) -> bond (L-2,L-1).

    At each bond (i, i+1):
    1. Form merged two-site tensor T = M_i . M_{i+1}.
    2. Evolve T forward by dt with H2_eff.
    3. SVD-split T (truncated); site i <- left factor (left-orthogonal).
    4. Evolve the right remnant SC backward by dt with H1_eff at site i+1.
    5. Set site i+1 <- SC_backward (the new centre tensor).
    6. Grow left environment from the new left-orthogonal site i tensor.
    """
    L = len(mps)
    for i in range(L - 1):
        M_i   = _get(mps, i)
        M_ip1 = _get(mps, i + 1)
        L_i   = L_cache[i]
        W_i   = mpo.tensors[i].data
        W_ip1 = mpo.tensors[i + 1].data
        R_ip2 = R_env[i + 2]

        # Merge
        T = np.tensordot(M_i, M_ip1, axes=([2], [0]))  # (chiL, d1, d2, chiR)

        # Forward two-site evolution
        T_new = _twosite_forward(T, L_i, W_i, W_ip1, R_ip2, dt, krylov_dim)

        # Truncated SVD split
        A_left, SC = _svd_split_left_trunc(T_new, chi_max, svd_tol)
        _set(mps, i, A_left)

        # Update left environment
        L_cache[i + 1] = _update_left_env(L_i, A_left, W_i)

        # Backward one-site evolution of the right remnant SC
        chiC, d2, chiR = SC.shape
        L_ip1 = L_cache[i + 1]
        R_ip1 = R_env[i + 2]   # right of site i+1
        H1_ip1 = _build_heff_onesite(L_ip1, W_ip1, R_ip1)

        def matvec_ip1(v):
            return H1_ip1 @ v

        SC_back = _expm_krylov(
            matvec_ip1, SC.astype(complex), +1j * dt, krylov_dim
        ).reshape(chiC, d2, chiR)
        _set(mps, i + 1, SC_back)


def _twosite_sweep_left(
    mps: MPS,
    mpo: MPO,
    L_cache: list,
    R_cache: list,
    dt: complex,
    krylov_dim: int,
    chi_max: int,
    svd_tol: float,
) -> None:
    """Left half-sweep for 2TDVP: bond (L-2,L-1) -> bond (0,1).

    Mirrors the right sweep with right-orthogonal splits.
    """
    L = len(mps)
    for i in range(L - 2, -1, -1):
        M_i   = _get(mps, i)
        M_ip1 = _get(mps, i + 1)
        L_i   = L_cache[i]
        W_i   = mpo.tensors[i].data
        W_ip1 = mpo.tensors[i + 1].data
        R_ip2 = R_cache[i + 2]

        # Merge
        T = np.tensordot(M_i, M_ip1, axes=([2], [0]))

        # Forward two-site evolution
        T_new = _twosite_forward(T, L_i, W_i, W_ip1, R_ip2, dt, krylov_dim)

        # Truncated SVD split (right-orthogonal)
        SC, A_right = _svd_split_right_trunc(T_new, chi_max, svd_tol)
        _set(mps, i + 1, A_right)

        # Update right environment
        R_cache[i + 1] = _update_right_env(R_ip2, A_right, W_ip1)

        # Backward one-site evolution of the left remnant SC
        chiL, d1, chiC = SC.shape
        L_i_   = L_cache[i]
        R_i_   = R_cache[i + 1]
        H1_i   = _build_heff_onesite(L_i_, W_i, R_i_)

        def matvec_i(v):
            return H1_i @ v

        SC_back = _expm_krylov(
            matvec_i, SC.astype(complex), +1j * dt, krylov_dim
        ).reshape(chiL, d1, chiC)
        _set(mps, i, SC_back)


# ---------------------------------------------------------------------------
# Public drivers
# ---------------------------------------------------------------------------


def tdvp1(
    env: Environment,
    mpo: MPO,
    mps0: MPS,
    config: TDVPConfig,
) -> TDVPResult:
    """Finite-size 1-site TDVP (1TDVP) for MPS time evolution.

    1TDVP evolves the MPS within the manifold of fixed bond dimension,
    projecting the Schrödinger equation onto the tangent space at each step.
    For real-time evolution under a Hermitian Hamiltonian, norm and energy
    are conserved to machine precision (up to Krylov truncation error).
    Bond dimensions are strictly fixed; the algorithm cannot increase the
    expressive power of the MPS during evolution.

    Algorithm (per step, Strang splitting)
    ---------------------------------------
    1. Right half-sweep (dt/2):  for i = 0 ... L-1
       a. Evolve centre tensor M_i forward with exp(-i H1_eff dt/2).
       b. QR: M_i' = Q R; store Q at site i; grow L_cache[i+1].
       c. Evolve bond centre C = R backward with exp(+i H0_eff dt/2).
       d. Absorb C into M_{i+1}.
    2. Left half-sweep (dt/2):  for i = L-1 ... 0
       a. Evolve centre tensor M_i forward with exp(-i H1_eff dt/2).
       b. LQ: M_i' = L Q; store Q at site i; grow R_cache[i].
       c. Evolve bond centre C = L backward with exp(+i H0_eff dt/2).
       d. Absorb C into M_{i-1}.

    Parameters
    ----------
    env:
        Environment specifying L, d, boundary conditions, and truncation.
    mpo:
        MPO Hamiltonian; must satisfy env.validate_hamiltonian(mpo).
    mps0:
        Initial MPS; should ideally be in a canonical form.  The algorithm
        will work from a mixed-canonical state with centre at site 0.
    config:
        Time-evolution parameters (dt, num_steps, krylov_dim, verbose).

    Returns
    -------
    TDVPResult containing the final MPS, times, energies, norms, and
    bond_dims at each step.
    """
    env.validate_hamiltonian(mpo)
    if len(mps0) != mpo.L:
        raise ValueError(f"MPS length {len(mps0)} != MPO length {mpo.L}")
    L = len(mps0)
    if L < 2:
        raise ValueError("tdvp1 requires L >= 2.")

    mps = mps0.copy()
    dt  = config.dt
    kd  = config.krylov_dim

    t_arr: List[float] = [0.0]
    E_list: List[float] = [float(np.real(expectation_value_env(mps, mpo)))]
    norms: List[float]  = [float(np.real(mps.norm()))]
    bd_list: List[List[int]] = [list(mps.bond_dims)]

    for step in range(1, config.num_steps + 1):
        half_dt = 0.5 * dt

        # --- Right half-sweep ---
        R_env   = build_right_environments(mps, mpo)
        L_envs  = build_left_environments(mps, mpo)
        L_cache = [None] * (L + 1)
        L_cache[0] = L_envs[0]

        _onesite_sweep_right(mps, mpo, L_cache, R_env, half_dt, kd)

        # --- Left half-sweep ---
        R_env2  = build_right_environments(mps, mpo)
        R_cache = [None] * (L + 1)
        R_cache[L] = R_env2[L]

        _onesite_sweep_left(mps, mpo, L_cache, R_cache, half_dt, kd)

        # --- Diagnostics ---
        t_arr.append(step * float(np.real(dt)))
        E_list.append(float(np.real(expectation_value_env(mps, mpo))))
        norms.append(float(np.real(mps.norm())))
        bd_list.append(list(mps.bond_dims))

        if config.verbose:
            print(
                f"[tdvp1] step {step:4d}:  t = {t_arr[-1]:.4f}  "
                f"E = {E_list[-1]:.10f}  norm = {norms[-1]:.6e}  "
                f"chi_max = {max(mps.bond_dims)}"
            )

    return TDVPResult(
        mps=mps,
        times=np.array(t_arr),
        energies=E_list,
        norms=norms,
        bond_dims=bd_list,
    )


def tdvp2(
    env: Environment,
    mpo: MPO,
    mps0: MPS,
    config: TDVPConfig,
    truncation: Optional[TruncationPolicy] = None,
    svd_tol: float = 0.0,
) -> TDVPResult:
    """Finite-size 2-site TDVP (2TDVP) for MPS time evolution.

    2TDVP merges adjacent pairs of site tensors, evolves the combined
    two-site object under the two-site effective Hamiltonian, and then
    splits the result via a truncated SVD.  This allows the bond dimension
    to grow adaptively during the evolution, at the cost of introducing
    truncation errors and breaking exact conservation of norm and energy.
    2TDVP is the preferred method when the initial bond dimension is
    insufficient to represent the growing entanglement of the evolving state.

    Algorithm (per step, Strang splitting)
    ---------------------------------------
    1. Right half-sweep (dt/2):  for i = 0 ... L-2
       a. Merge T = M_i . M_{i+1}.
       b. Evolve T forward with exp(-i H2_eff dt/2).
       c. Truncated SVD: T' = A_left . SC  (A_left left-orthogonal).
       d. Evolve SC backward with exp(+i H1_eff(i+1) dt/2) -> SC_back.
       e. Set site i <- A_left, site i+1 <- SC_back (new centre).
       f. Grow L_cache[i+1] from A_left.
    2. Left half-sweep (dt/2):  for i = L-2 ... 0
       a–f. Symmetric, using right-orthogonal SVD split.

    Parameters
    ----------
    env:
        Environment.
    mpo:
        MPO Hamiltonian.
    mps0:
        Initial MPS.
    config:
        Time-evolution parameters.
    truncation:
        Truncation policy for the SVD split.  Defaults to
        ``env.effective_truncation``.
    svd_tol:
        Relative singular-value cutoff for the SVD truncation (ratio to
        the largest singular value).  Values around 1e-8 to 1e-12 are
        typical.  Set to 0 to rely only on ``max_bond_dim``.

    Returns
    -------
    TDVPResult containing the final MPS, times, energies, norms, and
    bond_dims at each step.
    """
    env.validate_hamiltonian(mpo)
    if len(mps0) != mpo.L:
        raise ValueError(f"MPS length {len(mps0)} != MPO length {mpo.L}")
    L = len(mps0)
    if L < 2:
        raise ValueError("tdvp2 requires L >= 2.")

    if truncation is None:
        truncation = env.effective_truncation
    chi_max = truncation.max_bond_dim

    mps = mps0.copy()
    dt  = config.dt
    kd  = config.krylov_dim

    t_arr: List[float] = [0.0]
    E_list: List[float] = [float(np.real(expectation_value_env(mps, mpo)))]
    norms: List[float]  = [float(np.real(mps.norm()))]
    bd_list: List[List[int]] = [list(mps.bond_dims)]

    for step in range(1, config.num_steps + 1):
        half_dt = 0.5 * dt

        # --- Right half-sweep ---
        R_env   = build_right_environments(mps, mpo)
        L_envs  = build_left_environments(mps, mpo)
        L_cache = [None] * (L + 1)
        L_cache[0] = L_envs[0]

        _twosite_sweep_right(mps, mpo, L_cache, R_env, half_dt, kd, chi_max, svd_tol)

        # --- Left half-sweep ---
        R_env2  = build_right_environments(mps, mpo)
        R_cache = [None] * (L + 1)
        R_cache[L] = R_env2[L]

        _twosite_sweep_left(mps, mpo, L_cache, R_cache, half_dt, kd, chi_max, svd_tol)

        # --- Diagnostics ---
        t_arr.append(step * float(np.real(dt)))
        E_list.append(float(np.real(expectation_value_env(mps, mpo))))
        norms.append(float(np.real(mps.norm())))
        bd_list.append(list(mps.bond_dims))

        if config.verbose:
            print(
                f"[tdvp2] step {step:4d}:  t = {t_arr[-1]:.4f}  "
                f"E = {E_list[-1]:.10f}  norm = {norms[-1]:.6e}  "
                f"chi_max = {max(mps.bond_dims)}"
            )

    return TDVPResult(
        mps=mps,
        times=np.array(t_arr),
        energies=E_list,
        norms=norms,
        bond_dims=bd_list,
    )
