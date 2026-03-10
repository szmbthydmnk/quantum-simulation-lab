"""
Tests for Hamiltonian MPO builders.

Strategy: compare MPO.to_dense() against the independently-built
dense reference (tfim_dense, heisenberg_dense) for small chain lengths
L = 2, 3, 4. This catches FSM index errors, sign errors, and coupling
strength mistakes without needing any tensor-network machinery.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.core.hamiltonians import (
    tfim_mpo, tfim_dense,
    heisenberg_mpo, heisenberg_dense,
    xx_model_mpo,
    field_mpo,
)
from tensor_network_library.core.operators import (
    sigma_x, sigma_z,
    embed_operator, embed_two_site_operator,
    zz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_hermitian(M: np.ndarray, atol: float = 1e-12) -> bool:
    return np.allclose(M, M.conj().T, atol=atol)


# ---------------------------------------------------------------------------
# TFIM
# ---------------------------------------------------------------------------

class TestTFIMMPO:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_tfim_mpo_matches_dense_default_params(self, L):
        mpo_dense = tfim_mpo(L=L, J=1.0, g=1.0).to_dense()
        ref = tfim_dense(L=L, J=1.0, g=1.0)
        np.testing.assert_allclose(mpo_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_tfim_mpo_matches_dense_custom_params(self, L):
        J, g = 0.5, 1.7
        mpo_dense = tfim_mpo(L=L, J=J, g=g).to_dense()
        ref = tfim_dense(L=L, J=J, g=g)
        np.testing.assert_allclose(mpo_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_tfim_is_hermitian(self, L):
        H = tfim_mpo(L=L).to_dense()
        assert _is_hermitian(H)

    def test_tfim_bond_dims(self):
        mpo = tfim_mpo(L=4)
        assert mpo.bond_dims == [1, 3, 3, 3, 1]

    def test_tfim_pure_field_no_coupling(self):
        L, g = 3, 2.0
        H = tfim_mpo(L=L, J=0.0, g=g).to_dense()
        H_ref = sum(
            -g * embed_operator(sigma_x(), site=i, L=L, d=2)
            for i in range(L)
        )
        np.testing.assert_allclose(H, H_ref, atol=1e-12)

    def test_tfim_pure_ising_no_field(self):
        L, J = 3, 1.5
        H = tfim_mpo(L=L, J=J, g=0.0).to_dense()
        H_ref = sum(
            -J * embed_two_site_operator(zz(), site=i, L=L, d=2)
            for i in range(L - 1)
        )
        np.testing.assert_allclose(H, H_ref, atol=1e-12)

    def test_tfim_l_less_than_2_raises(self):
        with pytest.raises(ValueError):
            tfim_mpo(L=1)

    def test_tfim_shape(self):
        L = 3
        H = tfim_mpo(L=L).to_dense()
        assert H.shape == (2**L, 2**L)

    def test_tfim_l2_eigenvalues_match_dense(self):
        H = tfim_mpo(L=2, J=1.0, g=1.0).to_dense()
        evals = np.sort(np.linalg.eigvalsh(H))
        evals_ref = np.sort(np.linalg.eigvalsh(tfim_dense(L=2, J=1.0, g=1.0)))
        np.testing.assert_allclose(evals, evals_ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Heisenberg / XXZ
# ---------------------------------------------------------------------------

class TestHeisenbergMPO:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_heisenberg_mpo_matches_dense_isotropic(self, L):
        mpo_dense = heisenberg_mpo(L=L, Jx=1.0, Jy=1.0, Jz=1.0).to_dense()
        ref = heisenberg_dense(L=L, Jx=1.0, Jy=1.0, Jz=1.0)
        np.testing.assert_allclose(mpo_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_heisenberg_mpo_matches_dense_xxz(self, L):
        mpo_dense = heisenberg_mpo(L=L, Jx=1.0, Jy=1.0, Jz=0.5).to_dense()
        ref = heisenberg_dense(L=L, Jx=1.0, Jy=1.0, Jz=0.5)
        np.testing.assert_allclose(mpo_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_heisenberg_mpo_matches_dense_with_field(self, L):
        mpo_dense = heisenberg_mpo(L=L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.5).to_dense()
        ref = heisenberg_dense(L=L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.5)
        np.testing.assert_allclose(mpo_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_heisenberg_is_hermitian(self, L):
        H = heisenberg_mpo(L=L).to_dense()
        assert _is_hermitian(H)

    def test_heisenberg_bond_dims(self):
        mpo = heisenberg_mpo(L=4)
        assert mpo.bond_dims == [1, 5, 5, 5, 1]

    def test_heisenberg_zero_couplings_gives_field_only(self):
        L, h = 3, 1.0
        H = heisenberg_mpo(L=L, Jx=0.0, Jy=0.0, Jz=0.0, h=h).to_dense()
        H_ref = sum(
            -h * embed_operator(sigma_z(), site=i, L=L, d=2)
            for i in range(L)
        )
        np.testing.assert_allclose(H, H_ref, atol=1e-12)

    def test_heisenberg_l_less_than_2_raises(self):
        with pytest.raises(ValueError):
            heisenberg_mpo(L=1)

    def test_heisenberg_shape(self):
        L = 3
        H = heisenberg_mpo(L=L).to_dense()
        assert H.shape == (2**L, 2**L)


# ---------------------------------------------------------------------------
# XX Model
# ---------------------------------------------------------------------------

class TestXXModelMPO:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_xx_mpo_matches_heisenberg_jz0(self, L):
        xx_dense = xx_model_mpo(L=L, J=1.0).to_dense()
        ref = heisenberg_dense(L=L, Jx=1.0, Jy=1.0, Jz=0.0, h=0.0)
        np.testing.assert_allclose(xx_dense, ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_xx_is_hermitian(self, L):
        H = xx_model_mpo(L=L).to_dense()
        assert _is_hermitian(H)


# ---------------------------------------------------------------------------
# Field MPO
# ---------------------------------------------------------------------------

class TestFieldMPO:
    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_field_z_matches_sum_of_sigma_z(self, L):
        h = 1.3
        H = field_mpo(L=L, h=h, direction="z").to_dense()
        H_ref = sum(
            -h * embed_operator(sigma_z(), site=i, L=L, d=2)
            for i in range(L)
        )
        np.testing.assert_allclose(H, H_ref, atol=1e-12)

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_field_x_matches_sum_of_sigma_x(self, L):
        h = 0.7
        H = field_mpo(L=L, h=h, direction="x").to_dense()
        H_ref = sum(
            -h * embed_operator(sigma_x(), site=i, L=L, d=2)
            for i in range(L)
        )
        np.testing.assert_allclose(H, H_ref, atol=1e-12)

    def test_field_bond_dim_is_1(self):
        mpo = field_mpo(L=4, h=1.0)
        assert mpo.bond_dims == [1, 1, 1, 1, 1]

    def test_field_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            field_mpo(L=3, direction="w")

    @pytest.mark.parametrize("L", [2, 3, 4])
    def test_field_is_hermitian(self, L):
        H = field_mpo(L=L).to_dense()
        assert _is_hermitian(H)
