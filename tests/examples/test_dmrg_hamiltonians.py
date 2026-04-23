"""Tests for example DMRG Hamiltonian builders.

These tests validate that the simple example Hamiltonian MPOs produce
expectation values consistent with dense reference Hamiltonians, and
that the zz+z model matches the Heisenberg dense builder.
"""

from __future__ import annotations

import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.hamiltonian.operators import sigma_z, embed_operator
from tensor_network_library.hamiltonian.models import (
    heisenberg_dense, 
    random_field_mpo,
    xxz_mpo,
    xxz_dense,
    transverse_heisenberg_mpo,
    transverse_heisenberg_dense,
)
from examples.dmrg_hamiltonians import (
    random_z_field_mpo,
    random_x_field_mpo,
    zz_plus_z_mpo,
)


def random_normalized_state(L: int, *, dtype=np.complex128) -> np.ndarray:
    dim = 2 ** L
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v.astype(dtype)
    v /= np.linalg.norm(v)
    return v


class TestExampleHamiltonians:
    def test_random_z_field_matches_dense(self):
        L = 4
        J = np.array([0.7, 1.1, 0.9, 1.3], dtype=float)

        mpo = random_field_mpo(L=L, coefficients=J, direction="z")

        # Dense reference: sum_j J_j Z_j
        d = 2
        Z = sigma_z()
        H_dense = np.zeros((d**L, d**L), dtype=np.complex128)
        for j in range(L):
            H_dense += embed_operator(J[j] * Z, site=j, L=L, d=d)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = np.real(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)

    def test_zz_plus_z_matches_dense(self):
        L = 4
        Jz, h = 1.2, 0.5

        mpo = zz_plus_z_mpo(L=L, Jz=Jz, h=h)

        H_dense = heisenberg_dense(L=L, Jx=0.0, Jy=0.0, Jz=Jz, h=h)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = np.real(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)


class TestAdditionalHamiltonians:
    def test_xxz_matches_dense(self):
        """XXZ MPO expectation matches the dense XXZ Hamiltonian."""
        L = 6
        J = 1.0
        Delta = 0.7
        h = 0.2

        mpo = xxz_mpo(L = L, J = J, Delta = Delta, h = h)
        H_dense = xxz_dense(L = L, J = J, Delta = Delta, h = h)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = np.real(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)

    def test_transverse_heisenberg_matches_dense(self):
        """Transverse Heisenberg MPO expectation matches dense reference."""
        L = 3
        J = 4/3
        h = 1/3

        mpo = transverse_heisenberg_mpo(L=L, J=J, h=h)
        H_dense = transverse_heisenberg_dense(L=L, J=J, h=h)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = np.real(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)