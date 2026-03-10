"""
Hamiltonian MPO builders for standard 1D quantum lattice models.

All builders return an MPO whose to_dense() matches the dense Hamiltonian
for small L (use as a validation reference).

MPO finite-state-machine (FSM) structure used throughout:
  - Bond states encode partial sums of interaction terms.
  - Boundary bond dimensions are always 1.
  - Internal bond dimension is model-dependent (see each builder).

Canonical axis ordering per MPO site tensor:
    axis 0: left bond
    axis 1: physical in  (bra index)
    axis 2: physical out (ket index)
    axis 3: right bond
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .mpo import MPO
from .operators import sigma_x, sigma_y, sigma_z, identity

ComplexArray = NDArray[np.complex128]


# ---------------------------------------------------------------------------
# Internal FSM helper
# ---------------------------------------------------------------------------

def _set_fsm_tensor(
    W: np.ndarray,
    row: int,
    col: int,
    op: ComplexArray,
) -> None:
    """
    Set W[row, :, :, col] = op.

    W has shape (chi_left, d, d, chi_right).
    """
    W[row, :, :, col] = op


# ---------------------------------------------------------------------------
# Transverse-Field Ising Model (TFIM)
# ---------------------------------------------------------------------------

def tfim_mpo(
    L: int,
    J: float = 1.0,
    g: float = 1.0,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for the Transverse-Field Ising Model (TFIM):

        H = -J Σ_{i} σ_z^i σ_z^{i+1}  -  g Σ_i σ_x^i

    Bond dimension: chi = 3 for all internal bonds.
    FSM row/col encoding:
        0: identity pass-through (right edge)
        1: σ_z  (left operator of ZZ term, waiting for right partner)
        2: identity pass-through (left edge / accumulator)

    So W has shape (3, d, d, 3):
        W[2, :, :, 2] = I          (pass identity through middle)
        W[2, :, :, 1] = -J * σ_z  (start a ZZ term)
        W[2, :, :, 0] = -g * σ_x  (on-site transverse field)
        W[1, :, :, 0] = σ_z       (complete ZZ term)
        W[0, :, :, 0] = I         (pass identity to the left)

    Args:
        L: Chain length (must be >= 2).
        J: ZZ coupling strength.
        g: Transverse field strength.
        dtype: Tensor data type.

    Returns:
        MPO representing the TFIM Hamiltonian.
    """
    if L < 2:
        raise ValueError(f"TFIM requires L >= 2, got L={L}")

    d = 2
    chi = 3
    I  = identity(d, dtype)
    Sx = sigma_x(dtype)
    Sz = sigma_z(dtype)

    bond_dims = [1] + [chi] * (L - 1) + [1]
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)

    for i in range(L):
        W = np.zeros((bond_dims[i], d, d, bond_dims[i + 1]), dtype=dtype)

        left_dim  = bond_dims[i]    # 1 at i=0, else 3
        right_dim = bond_dims[i+1]  # 3 at i<L-1, else 1

        if L == 1:
            # Edge case: single-site chain (only transverse field)
            W[0, :, :, 0] = -g * Sx

        elif i == 0:
            # Left boundary: shape (1, d, d, 3)
            # Row 0 (only row): emit I, -J*Sz, -g*Sx into cols 2, 1, 0
            W[0, :, :, 2] = I
            W[0, :, :, 1] = -J * Sz
            W[0, :, :, 0] = -g * Sx

        elif i == L - 1:
            # Right boundary: shape (3, d, d, 1)
            # Col 0 (only col): receive from rows 2, 1, 0
            W[2, :, :, 0] = -g * Sx
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I

        else:
            # Bulk: shape (3, d, d, 3)
            W[2, :, :, 2] = I
            W[2, :, :, 1] = -J * Sz
            W[2, :, :, 0] = -g * Sx
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I

        mpo.tensors[i].data = W

    return mpo


# ---------------------------------------------------------------------------
# Heisenberg Model
# ---------------------------------------------------------------------------

def heisenberg_mpo(
    L: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for the XXZ Heisenberg model:

        H = Σ_i ( Jx σ_x^i σ_x^{i+1}
                + Jy σ_y^i σ_y^{i+1}
                + Jz σ_z^i σ_z^{i+1} )
          - h Σ_i σ_z^i

    Setting Jx=Jy=Jz=J gives the isotropic Heisenberg (XXX) model.
    Setting Jx=Jy=J, Jz=Delta gives the XXZ model.

    Bond dimension: chi = 5 for all internal bonds.
    FSM encoding (5 states):
        0: I  accumulator (left edge)
        1: Sz  (waiting for right Sz partner)
        2: Sy  (waiting for right Sy partner)
        3: Sx  (waiting for right Sx partner)
        4: I  pass-through (right edge)

    Args:
        L:  Chain length (must be >= 2).
        Jx: XX coupling.
        Jy: YY coupling.
        Jz: ZZ coupling.
        h:  Longitudinal field (-h σ_z per site).
        dtype: Tensor dtype.

    Returns:
        MPO for the Heisenberg Hamiltonian.
    """
    if L < 2:
        raise ValueError(f"Heisenberg MPO requires L >= 2, got L={L}")

    d = 2
    chi = 5
    I  = identity(d, dtype)
    Sx = sigma_x(dtype)
    Sy = sigma_y(dtype)
    Sz = sigma_z(dtype)

    bond_dims = [1] + [chi] * (L - 1) + [1]
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)

    for i in range(L):
        W = np.zeros((bond_dims[i], d, d, bond_dims[i + 1]), dtype=dtype)

        if i == 0:
            # Left boundary: shape (1, d, d, 5)
            W[0, :, :, 4] = I
            W[0, :, :, 3] = Jx * Sx
            W[0, :, :, 2] = Jy * Sy
            W[0, :, :, 1] = Jz * Sz
            W[0, :, :, 0] = -h * Sz

        elif i == L - 1:
            # Right boundary: shape (5, d, d, 1)
            W[4, :, :, 0] = -h * Sz
            W[3, :, :, 0] = Sx
            W[2, :, :, 0] = Sy
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I

        else:
            # Bulk: shape (5, d, d, 5)
            W[4, :, :, 4] = I
            W[4, :, :, 3] = Jx * Sx
            W[4, :, :, 2] = Jy * Sy
            W[4, :, :, 1] = Jz * Sz
            W[4, :, :, 0] = -h * Sz
            W[3, :, :, 0] = Sx
            W[2, :, :, 0] = Sy
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I

        mpo.tensors[i].data = W

    return mpo


# ---------------------------------------------------------------------------
# XX Model
# ---------------------------------------------------------------------------

def xx_model_mpo(
    L: int,
    J: float = 1.0,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for the XX model:

        H = J Σ_i ( σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} )

    This is the Heisenberg model with Jx=Jy=J, Jz=0, h=0.

    Bond dimension: chi = 4.

    Args:
        L: Chain length.
        J: Hopping/coupling strength.
        dtype: Tensor dtype.

    Returns:
        MPO for the XX Hamiltonian.
    """
    return heisenberg_mpo(L=L, Jx=J, Jy=J, Jz=0.0, h=0.0, dtype=dtype)


# ---------------------------------------------------------------------------
# Free longitudinal field
# ---------------------------------------------------------------------------

def field_mpo(
    L: int,
    h: float = 1.0,
    direction: str = "z",
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for a uniform single-site field:

        H = -h Σ_i σ_direction^i

    No nearest-neighbour coupling; bond dimension = 1.

    Args:
        L:         Chain length.
        h:         Field strength.
        direction: 'x', 'y', or 'z'.
        dtype:     Tensor dtype.

    Returns:
        MPO for the field Hamiltonian.
    """
    d = 2
    ops = {"x": sigma_x, "y": sigma_y, "z": sigma_z}
    if direction not in ops:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
    op = -h * ops[direction](dtype)

    bond_dims = [1] * (L + 1)
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)
    for i in range(L):
        W = np.zeros((1, d, d, 1), dtype=dtype)
        W[0, :, :, 0] = op
        mpo.tensors[i].data = W
    return mpo


# ---------------------------------------------------------------------------
# Dense reference builders (for testing only)
# ---------------------------------------------------------------------------

def tfim_dense(
    L: int,
    J: float = 1.0,
    g: float = 1.0,
    dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Build the TFIM Hamiltonian as a dense (d^L, d^L) matrix.

    Intended for validation against tfim_mpo().to_dense() for small L.

    Args:
        L: Chain length.
        J: ZZ coupling.
        g: Transverse field.
        dtype: Output dtype.

    Returns:
        Dense Hamiltonian matrix of shape (2^L, 2^L).
    """
    from .operators import embed_operator, embed_two_site_operator, zz

    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=dtype)

    # ZZ terms
    for i in range(L - 1):
        H -= J * embed_two_site_operator(zz(dtype), site=i, L=L, d=2, dtype=dtype)

    # Transverse field terms
    for i in range(L):
        H -= g * embed_operator(sigma_x(dtype), site=i, L=L, d=2, dtype=dtype)

    return H


def heisenberg_dense(
    L: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Build the Heisenberg Hamiltonian as a dense (d^L, d^L) matrix.

    Intended for validation against heisenberg_mpo().to_dense() for small L.

    Args:
        L:  Chain length.
        Jx: XX coupling.
        Jy: YY coupling.
        Jz: ZZ coupling.
        h:  Longitudinal field.
        dtype: Output dtype.

    Returns:
        Dense Hamiltonian matrix of shape (2^L, 2^L).
    """
    from .operators import embed_operator, embed_two_site_operator, xx, yy, zz

    dim = 2 ** L
    H = np.zeros((dim, dim), dtype=dtype)

    for i in range(L - 1):
        H += Jx * embed_two_site_operator(xx(dtype), site=i, L=L, d=2, dtype=dtype)
        H += Jy * embed_two_site_operator(yy(dtype), site=i, L=L, d=2, dtype=dtype)
        H += Jz * embed_two_site_operator(zz(dtype), site=i, L=L, d=2, dtype=dtype)

    for i in range(L):
        H -= h * embed_operator(sigma_z(dtype), site=i, L=L, d=2, dtype=dtype)

    return H
