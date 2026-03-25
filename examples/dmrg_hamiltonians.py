"""Simple Hamiltonian builders for DMRG examples.

This module provides three small Hamiltonian families as MPOs that are
useful for testing and demonstrating DMRG:

* H1 = sum_j J_j Z_j with random J_j ~ N(1, 0.1)
* H2 = sum_j J_j X_j with the same J_j distribution
* H3 = Jz sum_i Z_i Z_{i+1} - h sum_i Z_i (Ising-like ZZ+Z model)

The random field Hamiltonians H1 and H2 are now convenience wrappers
around :func:`tensor_network_library.hamiltonian.models.random_field_mpo`.
"""

from __future__ import annotations

import numpy as np

from tensor_network_library.core.mpo import MPO
from tensor_network_library.hamiltonian.operators import sigma_x, sigma_z
from tensor_network_library.hamiltonian.models import (
    heisenberg_mpo,
    random_field_mpo,
)


def random_z_field_mpo(
    L: int,
    *,
    mean: float = 1.0,
    var: float = 0.1,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """H1 = sum_j J_j Z_j with J_j ~ N(mean, var).

    This is a thin wrapper around :func:`random_field_mpo` with
    ``direction="z"`` and a freshly drawn array of coefficients.
    """
    J = np.random.normal(loc=mean, scale=np.sqrt(var), size=L)
    return random_field_mpo(L=L, coefficients=J, direction="z", dtype=dtype)


def random_x_field_mpo(
    L: int,
    *,
    mean: float = 1.0,
    var: float = 0.1,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """H2 = sum_j J_j X_j with J_j ~ N(mean, var).

    Thin wrapper around :func:`random_field_mpo` with ``direction="x"``.
    """
    J = np.random.normal(loc=mean, scale=np.sqrt(var), size=L)
    return random_field_mpo(L=L, coefficients=J, direction="x", dtype=dtype)


def zz_plus_z_mpo(
    L: int,
    *,
    Jz: float = 1.0,
    h: float = 0.5,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """H3 = Jz sum_i Z_i Z_{i+1} - h sum_i Z_i.

    This is obtained as a special case of the Heisenberg MPO with
    Jx = Jy = 0, only Jz and a longitudinal field h are non-zero.
    """
    return heisenberg_mpo(L=L, Jx=0.0, Jy=0.0, Jz=Jz, h=h, dtype=dtype)
