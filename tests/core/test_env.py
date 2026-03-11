"""
Tests for the Environment dataclass (core/env.py).
"""

import numpy as np
import pytest
from tensor_network_library.core.env import Environment
from tensor_network_library.core.policy import TruncationPolicy


class TestEnvironmentBasic:
    def test_construction(self):
        env = Environment(L=10, d=2)
        assert env.L == 10
        assert env.d == 2
        assert env.bc == "open"
        assert env.max_bond_dim == 64

    def test_hilbert_dim(self):
        env = Environment(L=4, d=2)
        assert env.hilbert_dim == 16

    def test_complex_dtype_normalised(self):
        env = Environment(L=4, d=2)
        assert env.complex_dtype == np.complex128

    def test_invalid_L_raises(self):
        with pytest.raises(ValueError):
            Environment(L=0, d=2)

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError):
            Environment(L=4, d=1)

    def test_invalid_bc_raises(self):
        with pytest.raises(ValueError):
            Environment(L=4, d=2, bc="helical")


class TestEnvironmentFactories:
    def test_qubit_chain(self):
        env = Environment.qubit_chain(L=8, chi_max=16)
        assert env.L == 8
        assert env.d == 2
        assert env.max_bond_dim == 16
        assert env.truncation is not None
        assert env.truncation.max_bond_dim == 16

    def test_spin1_chain(self):
        env = Environment.spin1_chain(L=6)
        assert env.d == 3
        assert env.truncation is None

    def test_qubit_chain_no_chi(self):
        env = Environment.qubit_chain(L=4)
        assert env.max_bond_dim == 64
        assert env.truncation is None


class TestEnvironmentEffectiveTruncation:
    def test_uses_explicit_truncation_first(self):
        trunc = TruncationPolicy(chi_max=32)
        env = Environment(L=4, d=2, truncation=trunc)
        assert env.effective_truncation is trunc

    def test_falls_back_to_max_bond_dim(self):
        env = Environment(L=4, d=2, max_bond_dim=16)
        t = env.effective_truncation
        assert t.max_bond_dim == 16


class TestEnvironmentRepr:
    def test_repr_contains_L_d(self):
        env = Environment(L=5, d=2)
        r = repr(env)
        assert "L=5" in r and "d=2" in r