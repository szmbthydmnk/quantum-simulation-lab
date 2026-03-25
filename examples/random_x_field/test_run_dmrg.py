"""Automated pytest tests for the H2 (random X-field) DMRG example.

These tests mirror the structure of the H1 tests so that both examples
are covered by the same kind of checks:

* MPO expectation value matches the dense reference.
* DMRG energies are monotonically non-increasing.
* Final DMRG energy converges to the exact ground-state energy.
* Bond-dimension bookkeeping is correct.
* run_dmrg.main() writes the expected output files.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.operators import sigma_x, embed_operator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

L_SMALL = 6
CHI_MAX = 8


def _fixed_x_field_mpo(L: int, J: np.ndarray) -> MPO:
    """Build H2 MPO with a deterministic array of couplings J."""
    X   = sigma_x()
    mpo = MPO.identity_mpo(L=L, d=2, dtype=np.complex128)
    for j in range(L):
        mpo.initialize_single_site_operator(J[j] * X, site=j)
    return mpo


def _dense_h2(L: int, J: np.ndarray) -> np.ndarray:
    """Dense H2 = sum_j J_j X_j as a full matrix."""
    d = 2
    X = sigma_x()
    H = np.zeros((d**L, d**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * X, site=j, L=L, d=d)
    return H


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestH2XFieldDMRG:
    """Tests for the random transverse X-field Hamiltonian example."""

    def test_mpo_expectation_matches_dense(self):
        """The MPO expectation value matches the dense reference for a random state."""
        rng = np.random.default_rng(10)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo     = _fixed_x_field_mpo(L, J)
        H_dense = _dense_h2(L, J)

        psi = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
        psi /= np.linalg.norm(psi)

        mps   = MPS.from_statevector(psi, physical_dims=2, normalize=True)
        e_mpo = expectation_value_env(mps, mpo)
        e_den = float(np.vdot(psi, H_dense @ psi).real)

        assert np.allclose(e_mpo, e_den, atol=1e-10), (
            f"MPO expectation {e_mpo:.12f} != dense {e_den:.12f}"
        )

    def test_analytic_ground_state_energy(self):
        """Exact ground-state energy of H2 equals -sum_j |J_j|."""
        rng = np.random.default_rng(11)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        H_dense      = _dense_h2(L, J)
        evals, _     = np.linalg.eigh(H_dense)
        E_exact      = float(evals[0])
        E_analytic   = -float(np.sum(np.abs(J)))

        assert abs(E_exact - E_analytic) < 1e-10, (
            f"Dense eigenvalue {E_exact:.12f} does not match -sum|J| = {E_analytic:.12f}"
        )

    def test_dmrg_energy_non_increasing(self):
        """DMRG energies are monotonically non-increasing across sweeps."""
        rng = np.random.default_rng(12)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo              = _fixed_x_field_mpo(L, J)
        evals, evecs     = np.linalg.eigh(_dense_h2(L, J))
        psi_gs           = evecs[:, 0]

        trunc  = TruncationPolicy(max_bond_dim=CHI_MAX)
        mps0   = MPS.from_statevector(psi_gs, physical_dims=2, truncation=trunc, normalize=True)
        env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
        config = DMRGConfig(max_sweeps=5, energy_tol=1e-12, verbose=False)
        result = finite_dmrg(env, mpo, mps0, config)

        energies = result.energies
        assert len(energies) >= 2
        for k in range(len(energies) - 1):
            assert energies[k + 1] <= energies[k] + 1e-10, (
                f"Energy increased at sweep {k}: {energies[k]:.12f} -> {energies[k+1]:.12f}"
            )

    def test_dmrg_matches_exact_ground_state(self):
        """Final DMRG energy matches the exact ground-state energy to 1e-8."""
        rng = np.random.default_rng(13)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo              = _fixed_x_field_mpo(L, J)
        evals, evecs     = np.linalg.eigh(_dense_h2(L, J))
        E_exact          = float(evals[0])
        psi_gs           = evecs[:, 0]

        trunc  = TruncationPolicy(max_bond_dim=CHI_MAX)
        mps0   = MPS.from_statevector(psi_gs, physical_dims=2, truncation=trunc, normalize=True)
        env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
        config = DMRGConfig(max_sweeps=10, energy_tol=1e-12, verbose=False)
        result = finite_dmrg(env, mpo, mps0, config)

        E_dmrg = result.energies[-1]
        assert abs(E_dmrg - E_exact) < 1e-8, (
            f"|E_dmrg - E_exact| = {abs(E_dmrg - E_exact):.3e} > 1e-8"
        )

    def test_dmrg_bond_dims_recorded(self):
        """DMRGResult.bond_dims has the correct structure."""
        rng = np.random.default_rng(14)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo              = _fixed_x_field_mpo(L, J)
        evals, evecs     = np.linalg.eigh(_dense_h2(L, J))
        psi_gs           = evecs[:, 0]

        trunc  = TruncationPolicy(max_bond_dim=CHI_MAX)
        mps0   = MPS.from_statevector(psi_gs, physical_dims=2, truncation=trunc, normalize=True)
        env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
        config = DMRGConfig(max_sweeps=3, energy_tol=1e-12, verbose=False)
        result = finite_dmrg(env, mpo, mps0, config)

        assert len(result.bond_dims) == len(result.energies)
        for bd in result.bond_dims:
            assert isinstance(bd, list)
            assert len(bd) == L + 1
            assert bd[0] == 1 and bd[-1] == 1

    def test_csv_and_plot_written(self, tmp_path: pathlib.Path, monkeypatch):
        """run_dmrg.main() writes the expected CSV and PNG files."""
        import examples.random_x_field.run_dmrg as script
        monkeypatch.setattr(script, "OUT_DIR", tmp_path)
        monkeypatch.setattr(script, "L", L_SMALL)
        monkeypatch.setattr(script, "CHI_MAX", CHI_MAX)
        monkeypatch.setattr(script, "MAX_SWEEPS", 3)

        script.main()

        assert (tmp_path / "H2_convergence.csv").exists()
        assert (tmp_path / "H2_energy_convergence.png").exists()
