"""
Tests for single-site and two-site operator primitives.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.core.operators import (
    identity,
    sigma_x, sigma_y, sigma_z,
    sigma_plus, sigma_minus,
    number_op,
    spin_x, spin_y, spin_z,
    two_site_op, xx, yy, zz, exchange,
    commutator, anticommutator,
    embed_operator, embed_two_site_operator,
)


# ---------------------------------------------------------------------------
# Single-site operators
# ---------------------------------------------------------------------------

class TestPauliBasicProperties:
    def test_sigma_x_hermitian(self):
        X = sigma_x()
        np.testing.assert_array_equal(X, X.conj().T)

    def test_sigma_y_hermitian(self):
        Y = sigma_y()
        np.testing.assert_array_equal(Y, Y.conj().T)

    def test_sigma_z_hermitian(self):
        Z = sigma_z()
        np.testing.assert_array_equal(Z, Z.conj().T)

    def test_sigma_x_values(self):
        np.testing.assert_array_equal(sigma_x(), [[0, 1], [1, 0]])

    def test_sigma_y_values(self):
        np.testing.assert_array_equal(sigma_y(), [[0, -1j], [1j, 0]])

    def test_sigma_z_values(self):
        np.testing.assert_array_equal(sigma_z(), [[1, 0], [0, -1]])

    def test_paulis_square_to_identity(self):
        I = identity()
        for op in [sigma_x(), sigma_y(), sigma_z()]:
            np.testing.assert_allclose(op @ op, I, atol=1e-14)

    def test_pauli_anticommutation(self):
        # {sigma_i, sigma_j} = 2 delta_ij I
        I = identity()
        X, Y, Z = sigma_x(), sigma_y(), sigma_z()
        np.testing.assert_allclose(anticommutator(X, Y), np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose(anticommutator(X, Z), np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose(anticommutator(Y, Z), np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose(anticommutator(X, X), 2 * I, atol=1e-14)

    def test_pauli_commutation_relations(self):
        # [X, Y] = 2iZ,  [Y, Z] = 2iX,  [Z, X] = 2iY
        X, Y, Z = sigma_x(), sigma_y(), sigma_z()
        np.testing.assert_allclose(commutator(X, Y), 2j * Z, atol=1e-14)
        np.testing.assert_allclose(commutator(Y, Z), 2j * X, atol=1e-14)
        np.testing.assert_allclose(commutator(Z, X), 2j * Y, atol=1e-14)

    def test_sigma_plus_minus_values(self):
        sp = sigma_plus()
        sm = sigma_minus()
        np.testing.assert_array_equal(sp, [[0, 1], [0, 0]])
        np.testing.assert_array_equal(sm, [[0, 0], [1, 0]])

    def test_sigma_plus_minus_are_conjugate_transposes(self):
        np.testing.assert_array_equal(sigma_plus(), sigma_minus().conj().T)

    def test_number_op_is_projector(self):
        n = number_op()
        np.testing.assert_allclose(n @ n, n, atol=1e-14)

    def test_number_op_relation_to_sigma_z(self):
        n = number_op()
        expected = (identity() - sigma_z()) / 2
        np.testing.assert_allclose(n, expected, atol=1e-14)

    def test_spin_ops_are_half_paulis(self):
        np.testing.assert_allclose(spin_x(), sigma_x() / 2, atol=1e-14)
        np.testing.assert_allclose(spin_y(), sigma_y() / 2, atol=1e-14)
        np.testing.assert_allclose(spin_z(), sigma_z() / 2, atol=1e-14)

    def test_identity_is_eye(self):
        np.testing.assert_array_equal(identity(2), np.eye(2))
        np.testing.assert_array_equal(identity(3), np.eye(3))


# ---------------------------------------------------------------------------
# Two-site operators
# ---------------------------------------------------------------------------

class TestTwoSiteOperators:
    def test_xx_shape(self):
        assert xx().shape == (4, 4)

    def test_xx_is_kron_of_sigma_x(self):
        expected = np.kron(sigma_x(), sigma_x())
        np.testing.assert_array_equal(xx(), expected)

    def test_yy_hermitian(self):
        Y2 = yy()
        np.testing.assert_allclose(Y2, Y2.conj().T, atol=1e-14)

    def test_zz_hermitian(self):
        Z2 = zz()
        np.testing.assert_allclose(Z2, Z2.conj().T, atol=1e-14)

    def test_exchange_is_sum_of_xx_yy_zz(self):
        np.testing.assert_allclose(exchange(), xx() + yy() + zz(), atol=1e-14)

    def test_two_site_op_generic(self):
        X, Z = sigma_x(), sigma_z()
        result = two_site_op(X, Z)
        np.testing.assert_array_equal(result, np.kron(X, Z))


# ---------------------------------------------------------------------------
# embed_operator
# ---------------------------------------------------------------------------

class TestEmbedOperator:
    def test_embed_sigma_z_site0_L2(self):
        result = embed_operator(sigma_z(), site=0, L=2, d=2)
        expected = np.kron(sigma_z(), identity())
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_embed_sigma_z_site1_L2(self):
        result = embed_operator(sigma_z(), site=1, L=2, d=2)
        expected = np.kron(identity(), sigma_z())
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_embed_sigma_x_middle_site_L3(self):
        result = embed_operator(sigma_x(), site=1, L=3, d=2)
        expected = np.kron(np.kron(identity(), sigma_x()), identity())
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_embed_out_of_range_raises(self):
        with pytest.raises(ValueError):
            embed_operator(sigma_z(), site=3, L=3, d=2)

    def test_embed_shape(self):
        result = embed_operator(sigma_z(), site=0, L=4, d=2)
        assert result.shape == (16, 16)


class TestEmbedTwoSiteOperator:
    def test_embed_zz_site0_L2(self):
        result = embed_two_site_operator(zz(), site=0, L=2, d=2)
        np.testing.assert_allclose(result, zz(), atol=1e-14)

    def test_embed_zz_site0_L3(self):
        result = embed_two_site_operator(zz(), site=0, L=3, d=2)
        expected = np.kron(zz(), identity())
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_embed_zz_site1_L3(self):
        result = embed_two_site_operator(zz(), site=1, L=3, d=2)
        expected = np.kron(identity(), zz())
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_embed_shape_L4(self):
        result = embed_two_site_operator(zz(), site=0, L=4, d=2)
        assert result.shape == (16, 16)

    def test_embed_out_of_range_raises(self):
        with pytest.raises(ValueError):
            embed_two_site_operator(zz(), site=2, L=3, d=2)
