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

from typing import Optional, Sequence
import numpy as np
from numpy.typing import NDArray

from ..core.mpo import MPO
from .operators import sigma_x, sigma_y, sigma_z, identity

ComplexArray = NDArray[np.complex128]


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
    MPO for the Transverse-Field Ising Model (TFIM)::

        H = -J Σ_{i} σ_z^i σ_z^{i+1}  -  g Σ_i σ_x^i

    Bond dimension: chi = 3 for all internal bonds.
    FSM row/col encoding:
        0: done (right vacuum)
        1: σ_z dangling (waiting for right ZZ partner)
        2: identity pass-through (left vacuum)

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

        if i == 0:
            W[0, :, :, 2] = I
            W[0, :, :, 1] = -J * Sz
            W[0, :, :, 0] = -g * Sx
        elif i == L - 1:
            W[2, :, :, 0] = -g * Sx
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I
        else:
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
    MPO for the XXZ Heisenberg model::

        H = Σ_i ( Jx σ_x^i σ_x^{i+1}
                + Jy σ_y^i σ_y^{i+1}
                + Jz σ_z^i σ_z^{i+1} )
          - h Σ_i σ_z^i

    Bond dimension: chi = 5 for all internal bonds.
    FSM encoding (5 states):
        0: done (right vacuum)
        1: Sz dangling
        2: Sy dangling
        3: Sx dangling
        4: identity pass-through (left vacuum)

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
            W[0, :, :, 4] = I
            W[0, :, :, 3] = Jx * Sx
            W[0, :, :, 2] = Jy * Sy
            W[0, :, :, 1] = Jz * Sz
            W[0, :, :, 0] = -h * Sz
        elif i == L - 1:
            W[4, :, :, 0] = -h * Sz
            W[3, :, :, 0] = Sx
            W[2, :, :, 0] = Sy
            W[1, :, :, 0] = Sz
            W[0, :, :, 0] = I
        else:
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
    MPO for the XX model::

        H = J Σ_i ( σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} )

    Thin wrapper over heisenberg_mpo with Jx=Jy=J, Jz=0, h=0.
    """
    return heisenberg_mpo(L=L, Jx=J, Jy=J, Jz=0.0, h=0.0, dtype=dtype)


# ---------------------------------------------------------------------------
# Uniform single-site field
# ---------------------------------------------------------------------------

def field_mpo(
    L: int,
    h: float = 1.0,
    direction: str = "z",
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for a uniform single-site field::

        H = -h Σ_i σ_direction^i

    Bond dimension chi=2 (FSM: identity pass-through + accumulate).

    Args:
        L:         Chain length.
        h:         Field strength (positive h lowers the energy along ``direction``).
        direction: ``'x'``, ``'y'``, or ``'z'``.
        dtype:     Tensor dtype.

    Returns:
        MPO for the uniform field Hamiltonian.
    """
    d = 2
    ops = {"x": sigma_x, "y": sigma_y, "z": sigma_z}
    if direction not in ops:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    op = -h * ops[direction](dtype)
    I  = identity(d, dtype)

    chi = 2
    bond_dims = [1] + [chi] * (L - 1) + [1]
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)

    for i in range(L):
        W = np.zeros((bond_dims[i], d, d, bond_dims[i + 1]), dtype=dtype)

        if i == 0:
            W[0, :, :, 1] = I
            W[0, :, :, 0] = op
        elif i == L - 1:
            W[1, :, :, 0] = op
            W[0, :, :, 0] = I
        else:
            W[1, :, :, 1] = I
            W[1, :, :, 0] = op
            W[0, :, :, 0] = I

        mpo.tensors[i].data = W

    return mpo


# ---------------------------------------------------------------------------
# Site-dependent single-site field (e.g. random transverse field)
# ---------------------------------------------------------------------------

def random_field_mpo(
    L: int,
    coefficients: Sequence[float],
    direction: str = "x",
    dtype: np.dtype = np.complex128,
) -> MPO:
    """
    MPO for a site-varying single-site field::

        H = Σ_i h_i σ_direction^i

    Uses the same chi=2 FSM as :func:`field_mpo`.

    FSM bond states:
        0 : ``done`` — all terms to the left of this site have been accumulated.
        1 : ``open`` — identity pass-through; no term has been emitted yet at
            this or any later site.

    Tensor structure per site i (bulk)::

        W[1, :, :, 1] = I            (pass identity through: not yet at this site)
        W[1, :, :, 0] = h_i * op    (emit local term and close)
        W[0, :, :, 0] = I            (propagate done state)

    Args:
        L:            Chain length.
        coefficients: Site-local field strengths ``h_i`` of length ``L``.
                      The Hamiltonian is ``H = Σ_i h_i * σ_direction^i``
                      (note: positive ``h_i`` raises energy; use negative
                      values for a ferromagnetic field).
        direction:    ``'x'``, ``'y'``, or ``'z'``.
        dtype:        Tensor dtype.

    Returns:
        MPO representing the site-varying field Hamiltonian.

    Raises:
        ValueError: if ``len(coefficients) != L`` or direction is invalid.
    """
    if len(coefficients) != L:
        raise ValueError(
            f"len(coefficients) = {len(coefficients)} must equal L = {L}"
        )
    ops = {"x": sigma_x, "y": sigma_y, "z": sigma_z}
    if direction not in ops:
        raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

    d   = 2
    op  = ops[direction](dtype)       # bare Pauli — coefficient applied per site
    I   = identity(d, dtype)
    chi = 2

    bond_dims = [1] + [chi] * (L - 1) + [1]
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)

    for i in range(L):
        h_i = complex(coefficients[i])
        W   = np.zeros((bond_dims[i], d, d, bond_dims[i + 1]), dtype=dtype)

        if i == 0:
            # Shape (1, d, d, 2)
            W[0, :, :, 1] = I
            W[0, :, :, 0] = h_i * op
        elif i == L - 1:
            # Shape (2, d, d, 1)
            W[1, :, :, 0] = h_i * op
            W[0, :, :, 0] = I
        else:
            # Shape (2, d, d, 2)
            W[1, :, :, 1] = I
            W[1, :, :, 0] = h_i * op
            W[0, :, :, 0] = I

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
    Dense TFIM Hamiltonian.  Used to validate ``tfim_mpo().to_dense()`` for small L.
    """
    from .operators import embed_operator, embed_two_site_operator, zz

    H = np.zeros((2**L, 2**L), dtype=dtype)
    for i in range(L - 1):
        H -= J * embed_two_site_operator(zz(dtype), site=i, L=L, d=2, dtype=dtype)
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
    Dense Heisenberg Hamiltonian.  Used to validate ``heisenberg_mpo().to_dense()``.
    """
    from .operators import embed_operator, embed_two_site_operator, xx, yy, zz

    H = np.zeros((2**L, 2**L), dtype=dtype)
    for i in range(L - 1):
        H += Jx * embed_two_site_operator(xx(dtype), site=i, L=L, d=2, dtype=dtype)
        H += Jy * embed_two_site_operator(yy(dtype), site=i, L=L, d=2, dtype=dtype)
        H += Jz * embed_two_site_operator(zz(dtype), site=i, L=L, d=2, dtype=dtype)
    for i in range(L):
        H -= h * embed_operator(sigma_z(dtype), site=i, L=L, d=2, dtype=dtype)
    return H


# ---------------------------------------------------------------------------
# Convenience wrappers: XXZ and transverse-field Heisenberg
# ---------------------------------------------------------------------------

def xxz_mpo(
        L: int,
        J: float = 1.0,
        Delta: float = 1.0,
        h: float = 0.0,
        dtype: np.dtype = np.complex128,
        ) -> MPO:
    """
    MPO for the XXZ model::

        H = J Σ_i ( σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} + Δ σ_z^i σ_z^{i+1} )
          - h Σ_i σ_z^i

    Thin wrapper over :func:`heisenberg_mpo`.
    """

    return heisenberg_mpo(L = L, Jx = J, Jy = J, Jz = Delta * J, h = h, dtype = dtype)

def xxz_dense(
        L: int,
        J: float = 1.0,
        Delta: float = 1.0,
        h: float = 0.0,
        dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Dense XXZ Hamiltonian via :func:'heisenberg_dense'.
    """

    return heisenberg_dense(L = L, Jx = J, Jy = J, Jz = Delta * J, h = h, dtype = dtype)

def transverse_heisenberg_mpo(
        L: int,
        J: float = 1.0,
        h: float = 0.0,
        dtype: np.dtype = np.complex128,
        ) -> MPO:
    """
    MPO for the Heisenberg model with a transverse (x) field::

        H = J Σ_i ( σ_x^i σ_x^{i+1}
                  + σ_y^i σ_y^{i+1}
                  + σ_z^i σ_z^{i+1} )
          - h Σ_i σ_x^i

    W-tensor convention (auxiliary bond dim = 5):

        Row/col index meaning
        ─────────────────────
        0  =  I  (left boundary / identity carry)
        1  =  J·Sz  (left leg of ZZ pair)
        2  =  J·Sy  (left leg of YY pair)
        3  =  J·Sx  (left leg of XX pair)
        4  =  I  (right boundary)

    At site 0 (left edge) the input bond dim is 1 so W has shape (1,d,d,5).
    At site L-1 (right edge) W has shape (5,d,d,1).
    """

    if L < 2:
        raise ValueError(f"Transverse Heisenberg MPO requires L >= 2, got L={L}")

    d = 2
    chi = 5
    I2 = identity(d, dtype)
    Sx = sigma_x(dtype)
    Sy = sigma_y(dtype)
    Sz = sigma_z(dtype)

    bond_dims = [1] + [chi] * (L - 1) + [1]
    mpo = MPO(L=L, d=d, bond_policy=bond_dims, dtype=dtype)

    for i in range(L):
        dl = bond_dims[i]   # left auxiliary bond dim
        dr = bond_dims[i + 1]  # right auxiliary bond dim
        W = np.zeros((dl, d, d, dr), dtype=dtype)

        if i == 0:
            # shape (1, d, d, 5)
            # Row 0 is the only input row.
            W[0, :, :, 0] = -h * Sx      # local field  → immediate output
            W[0, :, :, 1] = J * Sz       # start ZZ pair
            W[0, :, :, 2] = J * Sy       # start YY pair
            W[0, :, :, 3] = J * Sx       # start XX pair
            W[0, :, :, 4] = I2           # pass identity rightward

        elif i == L - 1:
            # shape (5, d, d, 1)
            W[0, :, :, 0] = I2           # identity from left boundary
            W[1, :, :, 0] = Sz           # close ZZ pair
            W[2, :, :, 0] = Sy           # close YY pair
            W[3, :, :, 0] = Sx           # close XX pair
            W[4, :, :, 0] = -h * Sx      # local field at rightmost site

        else:
            # shape (5, d, d, 5) — bulk sites
            W[0, :, :, 0] = I2           # pass left boundary identity
            W[1, :, :, 0] = Sz           # close ZZ pair
            W[2, :, :, 0] = Sy           # close YY pair
            W[3, :, :, 0] = Sx           # close XX pair
            W[4, :, :, 0] = -h * Sx      # local field
            W[4, :, :, 1] = J * Sz       # start new ZZ pair
            W[4, :, :, 2] = J * Sy       # start new YY pair
            W[4, :, :, 3] = J * Sx       # start new XX pair
            W[4, :, :, 4] = I2           # pass right boundary identity

        mpo.tensors[i].data = W

    return mpo


def transverse_heisenberg_dense(
    L: int,
    J: float = 1.0,
    h: float = 0.0,
    dtype: np.dtype = np.complex128,
) -> ComplexArray:
    """
    Dense transverse Heisenberg Hamiltonian.

    H = J Σ_i ( σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1} + σ_z^i σ_z^{i+1} )
      - h Σ_i σ_x^i
    """
    from .operators import embed_operator, embed_two_site_operator, xx, yy, zz

    H = np.zeros((2**L, 2**L), dtype=dtype)
    for i in range(L - 1):
        H += J * embed_two_site_operator(xx(dtype), site=i, L=L, d=2, dtype=dtype)
        H += J * embed_two_site_operator(yy(dtype), site=i, L=L, d=2, dtype=dtype)
        H += J * embed_two_site_operator(zz(dtype), site=i, L=L, d=2, dtype=dtype)
    for i in range(L):
        H -= h * embed_operator(sigma_x(dtype), site=i, L=L, d=2, dtype=dtype)
    return H
    
