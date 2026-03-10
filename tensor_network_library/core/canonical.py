"""Canonical form transformations for MPS.

Provides left-, right-, and mixed-canonicalization routines.
All routines operate on a copy of the input MPS and return the
canonicalized version; the original is not modified.

Axis convention per site tensor:
    axis 0 : left bond  (chi_left)
    axis 1 : physical   (d)
    axis 2 : right bond (chi_right)
"""

from __future__ import annotations

from typing import Optional
"""Canonical form transformations for MPS.

Provides left-, right-, and mixed-canonicalization routines.
All routines operate on a copy of the input MPS and return the
canonicalized version; the original is not modified.

Axis convention per site tensor:
    axis 0 : left bond  (chi_left)
    axis 1 : physical   (d)
    axis 2 : right bond (chi_right)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


from .mps import MPS
from .policy import TruncationPolicy
from .policy import TruncationPolicy
from .tensor import Tensor
from .index import Index


# ======================
# Internal helpers
# ======================


def _is_left_orthonormal(A: np.ndarray, 
                         atol: float = 1e-10,
                         ) -> bool:
    """
    Check that a single site tensor A satisfies left-orthonormality.

    A tensor A of shape (chi_L, d, chi_R) is left-orthonormal iff
        A^† A = I_{chi_R}   (contracting over chi_L and d)
    i.e. the matrix A.reshape(chi_L*d, chi_R) has orthonormal columns.
    """

    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    gram = M.conj().T @ M

    return bool(np.allclose(gram, np.eye(chi_R, dtype=A.dtype), atol=atol))

def _is_right_orthonormal(A: np.ndarray, 
                          atol: float = 1e-10,
                          ) -> bool:
    """
    Check that a single site tensor A satisfies right-orthonormality.

    A tensor A of shape (chi_L, d, chi_R) is right-orthonormal iff
        A A^† = I_{chi_L}   (contracting over d and chi_R)
    i.e. the matrix A.reshape(chi_L, d*chi_R) has orthonormal rows.
    """

    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L, d * chi_R)
    gram = M @ M.conj().T

    return bool(np.allclose(gram, np.eye(chi_L, dtype=A.dtype), atol=atol))

def _qr_left_step(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    QR-decompose site tensor A for a left-canonicalization sweep step.

    Args:
        A: Shape (chi_L, d, chi_R).

    Returns:
        Q: Shape (chi_L, d, chi_new) — left-orthonormal site tensor.
        R: Shape (chi_new, chi_R)    — upper-triangular remainder.
    """

    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L * d, chi_R)
    Q, R = np.linalg.qr(M, mode="reduced")
    chi_new = Q.shape[1]

    return Q.reshape(chi_L, d, chi_new), R


def _lq_right_step(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    LQ-decompose site tensor A for a right-canonicalization sweep step.

    We obtain the LQ factor by transposing and using QR:
        A.reshape(chi_L, d*chi_R) = L @ Q
    where Q has orthonormal rows (right-orthonormal side).

    Args:
        A: Shape (chi_L, d, chi_R).

    Returns:
        L: Shape (chi_L, chi_new)    — lower-triangular remainder.
        Q: Shape (chi_new, d, chi_R) — right-orthonormal site tensor.
    """

    chi_L, d, chi_R = A.shape
    M = A.reshape(chi_L, d * chi_R)
    # Transpose, QR, transpose back
    Qt, Lt = np.linalg.qr(M.conj().T, mode="reduced")
    # Qt: (d*chi_R, chi_new), Lt: (chi_new, chi_L)  ->  L: (chi_L, chi_new), Q: (chi_new, d*chi_R)
    L = Lt.conj().T          # (chi_L, chi_new)
    Q = Qt.conj().T          # (chi_new, d*chi_R)
    chi_new = Q.shape[0]

    return L, Q.reshape(chi_new, d, chi_R)


def _update_tensor(mps: MPS, 
                   site: int, 
                   data: np.ndarray,
                   ) -> None:
    
    """Replace the data of mps.tensors[site] and update its bond Index dims."""

    chi_L, d, chi_R = data.shape

    left_bond = Index(dim=chi_L, name=mps.bonds[site].name, tags=mps.bonds[site].tags)
    right_bond = Index(dim=chi_R, name=mps.bonds[site + 1].name, tags=mps.bonds[site + 1].tags)
    phys = mps.tensors[site].indices[1]  # physical index unchanged

    mps.bonds[site] = left_bond
    mps.bonds[site + 1] = right_bond
    mps.tensors[site] = Tensor(data, indices=[left_bond, phys, right_bond])


# ======================
# Public API
# ======================


def left_canonicalize(mps: MPS, 
                      policy: Optional[TruncationPolicy] = None,
                      ) -> MPS:
    """
    Bring MPS into left-canonical form with QR decomposition.
    As we do not need the singular value spectrum, QR is more efficient.
    
    In left-canonical form every site tensor A_i satisfies
        Σ_{s,α} A^s_{α,β} (A^s_{α,β'})* = δ_{β,β'}
    i.e. it is left-orthonormal.  The last site absorbs any overall scale.

    Args:
        mps:    Input MPS (not modified).
        policy: Optional truncation policy (reserved for future SVD-based variant).

    Returns:
        New MPS in left-canonical form.
    """

    mps = mps.copy()    # work with a copy to avoid modifying the input
    mps._assert_materialized()  # ensure we have the tensors in memory to work with
    
    for i  in range(mps.L - 1):
        # Following Schollwöck's notation
        A = mps.tensors[i].data          # (chi_L, d, chi_R)
        Q, R = _qr_left_step(A)          # Q: (chi_L, d, chi_new), R: (chi_new, chi_R)

        # Absorb R back into the MPS/next tensor:
        B = mps.tensors[i + 1].data     # (chi_R, d', chi_R')
        B_new = np.tensordot(R, B, axes = ([1], [0]))

        _update_tensor(mps, i, Q)
        _update_tensor(mps, i + 1, B_new)

    return mps


def right_canonicalize(mps: MPS,
                       policy: Optional[TruncationPolicy] = None,
                       ) -> MPS:
    """
    Bring an MPS into right-canonical form via LQ sweeps.

    In right-canonical form every site tensor B_i satisfies
        Σ_{s,β} B^s_{α,β} (B^s_{α',β})* = δ_{α,α'}
    i.e. it is right-orthonormal.  The first site absorbs any overall scale.

    Args:
        mps:    Input MPS (not modified).
        policy: Optional truncation policy (reserved for future SVD-based variant).

    Returns:
        New MPS in right-canonical form.
    """

    mps = mps.copy()
    mps._assert_materialized()

    for i in range(mps.L - 1, 0, -1):
        A = mps.tensors[i].data          # (chi_L, d, chi_R)
        L, Q = _lq_right_step(A)         # L: (chi_L, chi_new), Q: (chi_new, d, chi_R)

        # Absorb L into the previous tensor
        B = mps.tensors[i - 1].data      # (chi_L', d', chi_L)
        B_new = np.tensordot(B, L, axes=([2], [0]))  # (chi_L', d', chi_new)

        _update_tensor(mps, i, Q)
        _update_tensor(mps, i - 1, B_new)

    return mps


def mixed_canonicalize(mps: MPS,
                       center: int,
                       policy: Optional[TruncationPolicy] = None,
                       ) -> MPS:
    """
    Bring an MPS into mixed-canonical form with orthogonality center at `center`.

    Sites [0, center) are left-orthonormal.
    Sites (center, L) are right-orthonormal.
    Site `center` is the orthogonality center and is not constrained.

    Algorithm:
        1. Left-sweep QR from site 0 to center-1 (inclusive).
        2. Right-sweep LQ from site L-1 down to center+1 (inclusive).

    Args:
        mps:    Input MPS (not modified).
        center: Index of the orthogonality center (0 <= center < L).
        policy: Optional truncation policy (reserved for future SVD-based variant).

    Returns:
        New MPS in mixed-canonical form.

    Raises:
        ValueError: If `center` is out of range.
    """
    if not (0 <= center < mps.L):
        raise ValueError(
            f"center must satisfy 0 <= center < L={mps.L}, got center={center}"
        )

    mps = mps.copy()
    mps._assert_materialized()

    # --- Left sweep: sites 0 .. center-1 ---
    for i in range(center):
        A = mps.tensors[i].data
        Q, R = _qr_left_step(A)
        B = mps.tensors[i + 1].data
        B_new = np.tensordot(R, B, axes=([1], [0]))
        _update_tensor(mps, i, Q)
        _update_tensor(mps, i + 1, B_new)

    # --- Right sweep: sites L-1 .. center+1 ---
    for i in range(mps.L - 1, center, -1):
        A = mps.tensors[i].data
        L_mat, Q = _lq_right_step(A)
        B = mps.tensors[i - 1].data
        B_new = np.tensordot(B, L_mat, axes=([2], [0]))
        _update_tensor(mps, i, Q)
        _update_tensor(mps, i - 1, B_new)

    return mps


# ===========================
# Validation/External helpers
# ===========================


def is_left_orthonormal(A: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True if site tensor A (shape chi_L, d, chi_R) is left-orthonormal."""
    return _is_left_orthonormal(A, atol=atol)


def is_right_orthonormal(A: np.ndarray, atol: float = 1e-10) -> bool:
    """Return True if site tensor A (shape chi_L, d, chi_R) is right-orthonormal."""
    return _is_right_orthonormal(A, atol=atol)


















