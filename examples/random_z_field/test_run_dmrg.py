"""Automated pytest tests for the H1 (random Z-field) DMRG example.

These tests verify that the ``run_dmrg.py`` script and the core DMRG
algorithm produce correct results for a purely on-site Z-field
Hamiltonian H1 = sum_j J_j Z_j.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

# Make the repo root importable from any working directory.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.models import random_field_mpo
from tensor_network_library.hamiltonian.operators import sigma_z, embed_operator


L_SMALL   = 6
CHI_MAX   = 8


def _fixed_z_field_mpo(L: int, J: np.ndarray):
    """Build H1 MPO with a deterministic array of couplings J.

    Uses the core ``random_field_mpo`` builder with ``direction="z"``
    but passes a fixed J instead of sampling.
    """
    return random_field_mpo(L=L, coefficients=J, direction="z")


def _dense_h1(L: int, J: np.ndarray) -> np.ndarray:
    """Dense H1 = sum_j J_j Z_j as a full matrix."""
    d = 2
    Z = sigma_z()
    H = np.zeros((d**L, d**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * Z, site=j, L=L, d=d)
    return H


class TestH1ZFieldDMRG:
    """Tests for the random Z-field Hamiltonian example and core DMRG."""

    def test_mpo_expectation_matches_dense(self):
        """The MPO expectation value matches the dense reference for a random state."""
        rng = np.random.default_rng(0)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo     = _fixed_z_field_mpo(L, J)
        H_dense = _dense_h1(L, J)

        # Random normalised state
        psi = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
        psi /= np.linalg.norm(psi)

        mps    = MPS.from_statevector(psi, physical_dims=2, normalize=True)
        e_mpo  = expectation_value_env(mps, mpo)
        e_den  = float(np.vdot(psi, H_dense @ psi).real)

        assert np.allclose(e_mpo, e_den, atol=1e-10), (
            f"MPO expectation {e_mpo:.12f} != dense {e_den:.12f}"
        )

    def test_dmrg_energy_non_increasing(self):
        """DMRG energies are monotonically non-increasing across sweeps."""
        rng = np.random.default_rng(1)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo    = _fixed_z_field_mpo(L, J)
        evals, evecs = np.linalg.eigh(_dense_h1(L, J))
        psi_gs = evecs[:, 0]

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
        rng = np.random.default_rng(2)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo              = _fixed_z_field_mpo(L, J)
        evals, evecs     = np.linalg.eigh(_dense_h1(L, J))
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
        """DMRGResult.bond_dims is a list of lists with one entry per sweep."""
        rng = np.random.default_rng(3)
        L   = L_SMALL
        J   = rng.normal(1.0, np.sqrt(0.1), size=L)

        mpo    = _fixed_z_field_mpo(L, J)
        evals, evecs = np.linalg.eigh(_dense_h1(L, J))
        psi_gs = evecs[:, 0]

        trunc  = TruncationPolicy(max_bond_dim=CHI_MAX)
        mps0   = MPS.from_statevector(psi_gs, physical_dims=2, truncation=trunc, normalize=True)
        env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
        config = DMRGConfig(max_sweeps=3, energy_tol=1e-12, verbose=False)
        result = finite_dmrg(env, mpo, mps0, config)

        assert len(result.bond_dims) == len(result.energies)
        for bd in result.bond_dims:
            assert isinstance(bd, list)
            # L+1 bond dim entries, boundaries must be 1
            assert len(bd) == L + 1
            assert bd[0] == 1 and bd[-1] == 1

    def test_csv_and_plot_written(self, tmp_path: pathlib.Path, monkeypatch):
        """run_dmrg.main() writes the expected CSV and PNG files."""
        import examples.random_z_field.run_dmrg as script
        monkeypatch.setattr(script, "OUT_DIR", tmp_path)
        monkeypatch.setattr(script, "L", L_SMALL)
        monkeypatch.setattr(script, "CHI_MAX", CHI_MAX)
        monkeypatch.setattr(script, "MAX_SWEEPS", 3)

        script.main()

        assert (tmp_path / "H1_convergence.csv").exists()
        assert (tmp_path / "H1_energy_convergence.png").exists()
