"""Simple Hamiltonian builders for DMRG examples.

This module provides three small Hamiltonian families as MPOs that are
useful for testing and demonstrating DMRG:

* H1 = sum_j J_j Z_j with random J_j ~ N(1, 0.1)
* H2 = sum_j J_j X_j with the same J_j distribution
* H3 = Jz sum_i Z_i Z_{i+1} - h sum_i Z_i (Ising-like ZZ+Z model)

All builders return MPOs compatible with a qubit chain Environment,
using the existing MPO and operator infrastructure.
"""

from __future__ import annotations

import numpy as np

from tensor_network_library.core.mpo import MPO
from tensor_network_library.hamiltonian.operators import sigma_x, sigma_z
from tensor_network_library.hamiltonian.models import heisenberg_mpo


def random_z_field_mpo(
    L: int,
    *,
    mean: float = 1.0,
    var: float = 0.1,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """H1 = sum_j J_j Z_j with J_j ~ N(mean, var).

    Args:
        L:
            Chain length.
        mean, var:
            Mean and variance of the normal distribution used to draw the
            random couplings J_j.
        dtype:
            Complex dtype of the underlying tensors.

    Returns:
        An MPO representing the random longitudinal-field Hamiltonian.
    """
    d = 2
    mpo = MPO.identity_mpo(L=L, d=d, dtype=dtype)

    J = np.random.normal(loc=mean, scale=np.sqrt(var), size=L)
    Z = sigma_z(dtype)

    for j in range(L):
        op = J[j] * Z
        mpo.initialize_single_site_operator(op, site=j)

    return mpo


def random_x_field_mpo(
    L: int,
    *,
    mean: float = 1.0,
    var: float = 0.1,
    dtype: np.dtype = np.complex128,
) -> MPO:
    """H2 = sum_j J_j X_j with J_j ~ N(mean, var).

    Args:
        L:
            Chain length.
        mean, var:
            Mean and variance of the normal distribution used to draw the
            random couplings J_j.
        dtype:
            Complex dtype of the underlying tensors.

    Returns:
        An MPO representing the random transverse-field Hamiltonian.
    """
    d = 2
    mpo = MPO.identity_mpo(L=L, d=d, dtype=dtype)

    J = np.random.normal(loc=mean, scale=np.sqrt(var), size=L)
    X = sigma_x(dtype)

    for j in range(L):
        op = J[j] * X
        mpo.initialize_single_site_operator(op, site=j)

    return mpo


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

    Args:
        L:
            Chain length (must be >= 2 for the ZZ interaction).
        Jz:
            Strength of the nearest-neighbour ZZ coupling.
        h:
            Strength of the longitudinal field term (with a minus sign,
            matching the convention in :func:`heisenberg_mpo`).
        dtype:
            Complex dtype of the MPO tensors.

    Returns:
        An MPO representing the Ising-like ZZ+Z Hamiltonian.
    """
    return heisenberg_mpo(L=L, Jx=0.0, Jy=0.0, Jz=Jz, h=h, dtype=dtype)
