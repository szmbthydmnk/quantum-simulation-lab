"""Tests for the example DMRG Hamiltonian builders."""

import numpy as np

from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.core.mps import MPS
from tensor_network_library.hamiltonian.operators import (
    sigma_z,
    embed_operator,
)
from tensor_network_library.hamiltonian.models import heisenberg_dense
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

        # Build MPO with deterministic J by seeding and overriding
        rng = np.random.default_rng(123)
        _ = rng.normal  # just to clarify we're not using global state

        mpo = random_z_field_mpo(L=L, mean=0.0, var=1.0)
        # Overwrite with our fixed couplings for deterministic test
        Z = sigma_z()
        for j in range(L):
            mpo.initialize_single_site_operator(J[j] * Z, site=j)

        # Dense reference: sum_j J_j Z_j
        d = 2
        H_dense = np.zeros((d**L, d**L), dtype=np.complex128)
        for j in range(L):
            H_dense += embed_operator(J[j] * Z, site=j, L=L, d=d)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = float(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)

    def test_zz_plus_z_matches_dense(self):
        L = 4
        Jz, h = 1.2, 0.5

        mpo = zz_plus_z_mpo(L=L, Jz=Jz, h=h)

        H_dense = heisenberg_dense(L=L, Jx=0.0, Jy=0.0, Jz=Jz, h=h)

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        e_mpo = expectation_value_env(mps, mpo)
        e_dense = float(np.vdot(psi, H_dense @ psi))

        assert np.allclose(e_mpo, e_dense, atol=1e-10)
