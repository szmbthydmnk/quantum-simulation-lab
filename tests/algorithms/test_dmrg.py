"""
Tests for DMRG (single-site and two-site).

Strategy:
  - Single-site: energy of TFIM L=4 is within 0.1 of exact.
  - Two-site: energy of TFIM L=4 matches exact diagonalization within 1e-4.
  - Heisenberg L=4 2S: energy within 1e-4 of exact.
  - Convergence: energies list is non-increasing.
  - Invalid variant raises.
  - Environment integration: DMRG accepts an Environment and generates initial MPS.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.algorithms.dmrg import DMRG
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.env import Environment


def _tfim_dense(L: int, J: float = 1.0, g: float = 1.0) -> np.ndarray:
    d = 2
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    dim = d**L
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(L - 1):
        H -= J * np.kron(np.kron(np.eye(d**i), np.kron(Z, Z)), np.eye(d**(L-i-2)))
    for i in range(L):
        H -= g * np.kron(np.kron(np.eye(d**i), X), np.eye(d**(L-i-1)))
    return H


def _heisenberg_dense(L: int, J: float = 1.0) -> np.ndarray:
    d = 2
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    dim = d**L
    H = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(L - 1):
        for op in [X, Y, Z]:
            H += J * np.kron(np.kron(np.eye(d**i), np.kron(op, op)), np.eye(d**(L-i-2)))
    return H


class TestDMRGSingleSite:
    def test_tfim_energy_below_upper_bound(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        L = 4
        E_exact = float(np.linalg.eigvalsh(_tfim_dense(L))[0])
        mpo = tfim_mpo(L=L, J=1.0, g=1.0)
        dmrg = DMRG(mpo=mpo, variant="1S", n_sweeps=6, tol=1e-8)
        E, _ = dmrg.run()
        assert E < E_exact + 0.1

    def test_energies_non_increasing(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        mpo = tfim_mpo(L=4, J=1.0, g=1.0)
        dmrg = DMRG(mpo=mpo, variant="1S", n_sweeps=4)
        dmrg.run()
        for i in range(1, len(dmrg.energies)):
            assert dmrg.energies[i] <= dmrg.energies[i-1] + 1e-8

    def test_invalid_variant_raises(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        with pytest.raises(ValueError):
            DMRG(mpo=tfim_mpo(L=4), variant="3S")


class TestDMRGTwoSite:
    def test_tfim_energy_matches_exact(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        L = 4
        E_exact = float(np.linalg.eigvalsh(_tfim_dense(L))[0])
        mpo = tfim_mpo(L=L, J=1.0, g=1.0)
        trunc = TruncationPolicy(chi_max=16)
        dmrg = DMRG(mpo=mpo, variant="2S", truncation=trunc, n_sweeps=10, tol=1e-10)
        E, _ = dmrg.run()
        assert abs(E - E_exact) < 1e-4

    def test_heisenberg_energy_close_to_exact(self):
        from tensor_network_library.hamiltonian.models import heisenberg_mpo
        L = 4
        E_exact = float(np.linalg.eigvalsh(_heisenberg_dense(L))[0])
        mpo = heisenberg_mpo(L=L, Jx=1.0, Jy=1.0, Jz=1.0)
        trunc = TruncationPolicy(chi_max=16)
        dmrg = DMRG(mpo=mpo, variant="2S", truncation=trunc, n_sweeps=10, tol=1e-10)
        E, _ = dmrg.run()
        assert abs(E - E_exact) < 1e-4

    def test_ground_state_mps_is_normalized(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        mpo = tfim_mpo(L=4, J=1.0, g=1.0)
        trunc = TruncationPolicy(chi_max=8)
        dmrg = DMRG(mpo=mpo, variant="2S", truncation=trunc, n_sweeps=6)
        _, mps = dmrg.run()
        np.testing.assert_allclose(mps.norm(), 1.0, atol=1e-6)

    def test_environment_integration(self):
        from tensor_network_library.hamiltonian.models import tfim_mpo
        env = Environment.qubit_chain(L=4, chi_max=8)
        mpo = tfim_mpo(L=4)
        dmrg = DMRG(mpo=mpo, variant="2S", env=env, n_sweeps=4)
        E, _ = dmrg.run()
        assert np.isfinite(E)
