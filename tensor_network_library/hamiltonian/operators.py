"""
Single-site and two-site operator primitives.

All single-site operators are returned as (d, d) complex128 arrays.
Two-site operators are returned as (d*d, d*d) matrices in the
lexicographic basis |ij> = |i> ⊗ |j>, or as rank-4 tensors (d,d,d,d)
where axes are (bra_left, bra_right, ket_left, ket_right).

Convention for Pauli matrices follows the standard physics convention:
    σ_x = [[0,1],[1,0]],  σ_y = [[0,-i],[i,0]],  σ_z = [[1,0],[0,-1]]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Single-site operators
# ---------------------------------------------------------------------------

def identity(d: int = 2, dtype: np.dtype = np.complex128) -> ComplexArray:
    """d×d identity operator."""
    return np.eye(d, dtype=dtype)


def sigma_x(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Pauli-X (σ_x) for a qubit (d=2)."""
    return np.array([[0, 1],
                     [1, 0]], dtype=dtype)


def sigma_y(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Pauli-Y (σ_y) for a qubit (d=2)."""
    return np.array([[0, -1j],
                     [1j,  0]], dtype=dtype)


def sigma_z(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Pauli-Z (σ_z) for a qubit (d=2)."""
    return np.array([[1,  0],
                     [0, -1]], dtype=dtype)


def sigma_plus(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Raising operator σ_+ = |0><1| for a qubit."""
    return np.array([[0, 1],
                     [0, 0]], dtype=dtype)


def sigma_minus(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Lowering operator σ_- = |1><0| for a qubit."""
    return np.array([[0, 0],
                     [1, 0]], dtype=dtype)


def number_op(dtype: np.dtype = np.complex128) -> ComplexArray:
    """
    Number / excited-state projector n = |1><1| = (I - σ_z) / 2.

    Eigenvalues: 0 for |0>, 1 for |1>.
    """
    return np.array([[0, 0],
                     [0, 1]], dtype=dtype)


def spin_x(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Spin-1/2 operator S_x = σ_x / 2."""
    return sigma_x(dtype) / 2


def spin_y(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Spin-1/2 operator S_y = σ_y / 2."""
    return sigma_y(dtype) / 2


def spin_z(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Spin-1/2 operator S_z = σ_z / 2."""
    return sigma_z(dtype) / 2


# ---------------------------------------------------------------------------
# Two-site operators
# ---------------------------------------------------------------------------

def two_site_op(
    op_left: ComplexArray,
    op_right: ComplexArray,
) -> ComplexArray:
    """
    Tensor product of two single-site operators: op_left ⊗ op_right.

    Args:
        op_left:  (d, d) operator on left site.
        op_right: (d, d) operator on right site.

    Returns:
        (d*d, d*d) two-site operator in lexicographic basis.
    """
    return np.kron(op_left, op_right)


def xx(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Two-site σ_x ⊗ σ_x coupling (d=2), shape (4,4)."""
    return two_site_op(sigma_x(dtype), sigma_x(dtype))


def yy(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Two-site σ_y ⊗ σ_y coupling (d=2), shape (4,4)."""
    return two_site_op(sigma_y(dtype), sigma_y(dtype))


def zz(dtype: np.dtype = np.complex128) -> ComplexArray:
    """Two-site σ_z ⊗ σ_z coupling (d=2), shape (4,4)."""
    return two_site_op(sigma_z(dtype), sigma_z(dtype))


def exchange(dtype: np.dtype = np.complex128) -> ComplexArray:
    """
    Heisenberg exchange interaction: XX + YY + ZZ (d=2), shape (4,4).

    This equals 2(σ_+ ⊗ σ_- + σ_- ⊗ σ_+) + ZZ, and is proportional
    to the permutation operator for spin-1/2.
    """
    return xx(dtype) + yy(dtype) + zz(dtype)


# ---------------------------------------------------------------------------
# Operator algebra utilities
# ---------------------------------------------------------------------------

def commutator(A: ComplexArray, B: ComplexArray) -> ComplexArray:
    """Matrix commutator [A, B] = AB - BA."""
    return A @ B - B @ A


def anticommutator(A: ComplexArray, B: ComplexArray) -> ComplexArray:
    """Matrix anticommutator {A, B} = AB + BA."""
    return A @ B + B @ A


def embed_operator(
    op: ComplexArray,
    site: int,
    L: int,
    d: int = 2,
    dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Embed a single-site operator into the full L-site Hilbert space.

    Returns I ⊗ ... ⊗ op ⊗ ... ⊗ I of shape (d^L, d^L).
    Intended for small L only (dense construction for testing/validation).

    Args:
        op:   (d, d) single-site operator.
        site: Site index where op acts (0-indexed).
        L:    Chain length.
        d:    Local dimension.
        dtype: Output dtype.

    Returns:
        Full operator of shape (d**L, d**L).
    """
    if not (0 <= site < L):
        raise ValueError(f"site={site} out of range [0, {L})")
    if op.shape != (d, d):
        raise ValueError(f"op shape {op.shape} inconsistent with d={d}")

    result = np.array([[1.0 + 0j]], dtype=dtype)
    for i in range(L):
        result = np.kron(result, op.astype(dtype) if i == site else identity(d, dtype))
    return result


def embed_two_site_operator(
    op: ComplexArray,
    site: int,
    L: int,
    d: int = 2,
    dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Embed a two-site operator op acting on (site, site+1) into full space.

    Returns I ⊗ ... ⊗ op_{i,i+1} ⊗ ... ⊗ I of shape (d^L, d^L).
    Intended for small L only.

    Args:
        op:   (d*d, d*d) two-site operator.
        site: Left site index (right site = site+1).
        L:    Chain length.
        d:    Local dimension.
        dtype: Output dtype.

    Returns:
        Full operator of shape (d**L, d**L).
    """
    if not (0 <= site < L - 1):
        raise ValueError(f"site={site} out of range for two-site op on L={L} chain")
    if op.shape != (d * d, d * d):
        raise ValueError(f"op shape {op.shape} inconsistent with d={d} two-site op")

    left_dim  = d ** site
    right_dim = d ** (L - site - 2)

    result = np.kron(
        np.kron(identity(left_dim, dtype), op.astype(dtype)),
        identity(right_dim, dtype),
    )
    return result