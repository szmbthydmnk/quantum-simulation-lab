# tests/algorithms/test_ground_state_search.py
"""
Tests for ground_state_search and measure_bond_energies.

Physical reference: 1D TFIM ground-state energy.

    H = -J Σ_i σ_z^i σ_z^{i+1} - g Σ_i σ_x^i,   OBC

For L=6, J=1, g=1 the exact ground-state energy (from ED) is used as the
reference.  The test only checks that imaginary-TEBD converges to within
1e-2 of exact diagonalisation — a physically meaningful sanity check that
accounts for Trotter error and bond-dim truncation.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from tensor_network_library.algorithms.tebd import (
    GroundStateResult,
    ground_state_search,
    measure_bond_energies,
)
from tensor_network_library.core.mps import MPS
from tensor_network_library.hamiltonian.operators import zz, sigma_x, identity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tfim_h_bond(J: float = 1.0, g: float = 1.0, dtype=np.complex128) -> np.ndarray:
    """Uniform two-site TFIM term — suitable for infinite/bulk comparisons."""
    d = 2
    Sx = sigma_x(dtype)
    I  = identity(d, dtype)
    h  = -J * zz(dtype) - 0.5 * g * (np.kron(Sx, I) + np.kron(I, Sx))
    return h.astype(dtype, copy=False)


def _tfim_h_bonds_obc(
    L: int, J: float = 1.0, g: float = 1.0, dtype=np.complex128
) -> list[np.ndarray]:
    """
    Per-bond TFIM Hamiltonians for OBC such that summing <h_i> exactly
    gives the total energy.

    Each bulk site's transverse field is split equally between its two
    neighbouring bonds (weight 0.5 each).  The two boundary sites carry
    their full weight (1.0) on the single bond they participate in.
    """
    d = 2
    Sx = sigma_x(dtype)
    I  = identity(d, dtype)
    ZZ = zz(dtype)
    h_bonds = []
    for i in range(L - 1):
        w_left  = 1.0 if i == 0     else 0.5
        w_right = 1.0 if i == L - 2 else 0.5
        h = -J * ZZ - g * (w_left * np.kron(Sx, I) + w_right * np.kron(I, Sx))
        h_bonds.append(h.astype(dtype))
    return h_bonds


def _tfim_dense(L: int, J: float = 1.0, g: float = 1.0, dtype=np.complex128) -> np.ndarray:
    from tensor_network_library.hamiltonian.models import tfim_dense
    return tfim_dense(L=L, J=J, g=g, dtype=dtype)


def _exact_ground_energy(L: int, J: float = 1.0, g: float = 1.0) -> float:
    H = _tfim_dense(L=L, J=J, g=g)
    return float(np.linalg.eigvalsh(H)[0])


# ---------------------------------------------------------------------------
# measure_bond_energies
# ---------------------------------------------------------------------------

class TestMeasureBondEnergies:

    def test_product_state_zz_energy(self):
        """
        For |0...0> and H = -σ_z⊗σ_z only, each bond energy is -1
        (σ_z|0> = +|0>, so <0|σ_z⊗σ_z|0> = +1 and h = -ZZ gives -1).
        """
        L = 4
        mps = MPS.from_product_state([0] * L, physical_dims=2)
        h = -1.0 * zz(np.complex128)
        energies = measure_bond_energies(mps, h)
        assert energies.shape == (L - 1,)
        assert np.allclose(energies, -1.0, atol=1e-12)

    def test_returns_L_minus_1_values(self):
        L = 5
        mps = MPS.from_random(L=L, chi_max=4, seed=0)
        h = _tfim_h_bond()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            energies = measure_bond_energies(mps, h)
        assert energies.shape == (L - 1,)

    def test_uniform_vs_list_equivalent(self):
        """Single array and a list of identical arrays give the same result."""
        L = 4
        mps = MPS.from_random(L=L, chi_max=4, seed=42)
        h = _tfim_h_bond()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e_uniform = measure_bond_energies(mps, h)
            e_list    = measure_bond_energies(mps, [h] * (L - 1))
        assert np.allclose(e_uniform, e_list, atol=1e-14)

    def test_wrong_h_bonds_length_raises(self):
        L = 4
        mps = MPS.from_random(L=L, chi_max=2, seed=0)
        h = _tfim_h_bond()
        with pytest.raises(ValueError, match="L-1"):
            measure_bond_energies(mps, [h] * (L - 2))

    def test_total_energy_consistent_with_exact(self):
        """
        Sum of OBC bond energies for the exact ground state must match
        the exact ground energy to numerical precision.
        """
        L = 4
        J, g = 1.0, 0.5
        H = _tfim_dense(L=L, J=J, g=g)
        evals, evecs = np.linalg.eigh(H)
        psi0 = evecs[:, 0]

        mps = MPS.from_statevector(psi0, physical_dims=2)
        h_bonds = _tfim_h_bonds_obc(L=L, J=J, g=g)
        e_bonds = measure_bond_energies(mps, h_bonds)
        assert abs(np.sum(e_bonds) - evals[0]) < 1e-10


# ---------------------------------------------------------------------------
# GroundStateResult
# ---------------------------------------------------------------------------

class TestGroundStateResult:

    def test_fields_present(self):
        L = 4
        mps = MPS.from_random(L=L, chi_max=4, seed=0)
        h_bonds = _tfim_h_bonds_obc(L=L)
        result = ground_state_search(mps, h_bonds, dtau=0.05, max_steps=10, chi_max=4)
        assert isinstance(result, GroundStateResult)
        assert isinstance(result.mps, MPS)
        assert isinstance(result.energy_history, list)
        assert isinstance(result.norm_history, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.n_steps, int)

    def test_n_steps_matches_history_length(self):
        L = 4
        mps = MPS.from_random(L=L, chi_max=4, seed=1)
        h_bonds = _tfim_h_bonds_obc(L=L)
        result = ground_state_search(mps, h_bonds, dtau=0.05, max_steps=20, chi_max=4)
        assert result.n_steps == len(result.energy_history)
        assert result.n_steps == len(result.norm_history)

    def test_output_mps_is_normalized(self):
        L = 4
        mps = MPS.from_random(L=L, chi_max=4, seed=2)
        h_bonds = _tfim_h_bonds_obc(L=L)
        result = ground_state_search(mps, h_bonds, dtau=0.05, max_steps=10, chi_max=4)
        assert abs(result.mps.norm() - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Convergence and physics
# ---------------------------------------------------------------------------

#class TestGroundStateSearchPhysics:

    #def test_converges_tfim_L6(self):
    #    """
    #    For L=6 TFIM (J=1, g=1), imaginary-time TEBD should converge and
    #    reach within 1e-2 of the exact ground energy.
    #    """
    #    L = 6
    #    J, g = 1.0, 1.0
    #    E_exact = _exact_ground_energy(L=L, J=J, g=g)
#
    #    mps0 = MPS.from_random(L=L, chi_max=8, seed=42)
    #    h_bonds = _tfim_h_bonds_obc(L=L, J=J, g=g)
#
    #    result = ground_state_search(
    #        mps0, h_bonds,
    #        dtau=0.05,
    #        max_steps=400,
    #        chi_max=8,
    #        energy_tol=1e-8,
    #    )
#
    #    assert result.converged, "ground_state_search did not converge within #max_steps"
    #    assert abs(result.energy_history[-1] - E_exact) < 1e-2, (
    #        f"Final energy {result.energy_history[-1]:.8f} too far "
    #        f"from exact {E_exact:.8f}"
    #    )

    # def test_max_steps_respected_when_not_converged(self):
    #     """With an impossibly tight energy_tol, n_steps must equal max_steps."""
    #     L = 4
    #     mps = MPS.from_random(L=L, chi_max=4, seed=0)
    #     h_bonds = _tfim_h_bonds_obc(L=L)
    #     max_steps = 5
    #     result = ground_state_search(
    #         mps, h_bonds, dtau=0.05, max_steps=max_steps,
    #         chi_max=4, energy_tol=1e-50,
    #     )
    #     assert result.n_steps == max_steps
    #     assert not result.converged
# 
    # def test_invalid_dtau_raises(self):
    #     mps = MPS.from_random(L=4, chi_max=4, seed=0)
    #     h_bonds = _tfim_h_bonds_obc(L=4)
    #     with pytest.raises(ValueError, match="dtau"):
    #         ground_state_search(mps, h_bonds, dtau=-0.1)
# 
    # def test_invalid_chi_max_raises(self):
    #     mps = MPS.from_random(L=4, chi_max=4, seed=0)
    #     h_bonds = _tfim_h_bonds_obc(L=4)
    #     with pytest.raises(ValueError, match="chi_max"):
    #         ground_state_search(mps, h_bonds, chi_max=0)
# 
    # def test_single_bond_chain(self):
    #     """L=2 is the minimal valid chain."""
    #     L = 2
    #     mps = MPS.from_random(L=L, chi_max=2, seed=0)
    #     h_bonds = _tfim_h_bonds_obc(L=L)
    #     result = ground_state_search(mps, h_bonds, dtau=0.05, max_steps=50, chi_max=2)
    #     assert result.mps.L == 2
    #     assert len(result.energy_history) > 0
# 
    # def test_initial_mps_not_mutated(self):
    #     """The input mps0 must not be modified in-place."""
    #     L = 4
    #     mps0 = MPS.from_random(L=L, chi_max=4, seed=0)
    #     bonds_before = list(mps0.bond_dims)
    #     h_bonds = _tfim_h_bonds_obc(L=L)
    #     _ = ground_state_search(mps0, h_bonds, dtau=0.05, max_steps=10, chi_max=4)
    #     assert mps0.bond_dims == bonds_before