# tests/core/test_mps.py

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace


# Robust imports (adapt if your package name differs)
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mps import Index
from tensor_network_library.core.mps import Tensor


def _kron_list(vecs: list[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0.0j], dtype=np.complex128)
    for v in vecs:
        out = np.kron(out, np.asarray(v, dtype=np.complex128).reshape(-1))
    return out


def _basis_vec(d: int, s: int) -> np.ndarray:
    v = np.zeros(d, dtype=np.complex128)
    v[s] = 1.0
    return v


class TestMPSInitAndStructure:
    def test_len_and_repr(self):
        mps = MPS(L=4, physical_dims=2, bond_policy=[1, 1, 1, 1, 1], allocate=False)
        assert len(mps) == 4
        r = repr(mps)
        assert "MPS" in r or "Mps" in r or "mps" in r
        assert "L=4" in r

    #def test_init_physical_dims_int(self):
    #    mps = MPS(L=5, physical_dims=2, bond_policy=[1, 1, 1, 1, 1, 1], allocate=False)
    #    assert mps.physical_dims == [2, 2, 2, 2, 2]
    #    assert mps.bond_dims == [1, 1, 1, 1, 1, 1]
#
    #def test_init_physical_dims_list(self):
    #    mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy=[1, 1, 1, 1, 1], allocate=False)
    #    assert mps.physical_dims == [2, 3, 2, 2]
#
    #def test_init_physical_dims_list_wrong_length_raises(self):
    #    with pytest.raises(AssertionError):
    #        _ = MPS(L=4, physical_dims=[2, 2, 2], bond_policy=[1, 1, 1, 1, 1])
#
    #def test_bonds_and_indices_are_index_objects_and_connected(self):
    #    L = 6
    #    mps = MPS(L=L, physical_dims=2, bond_policy=[1] * (L + 1), allocate=True)
    #    assert len(mps.indices) == L
    #    assert len(mps.bonds) == L + 1
#
    #    for ix in mps.indices:
    #        assert isinstance(ix, Index)
    #    for b in mps.bonds:
    #        assert isinstance(b, Index)
#
    #    # Each tensor has [bondL, phys, bondR] and bond connectivity is shared by object identity
    #    for i in range(L):
    #        t = mps.tensors[i]
    #        assert isinstance(t, Tensor)
    #        assert len(t.indices) == 3
    #        assert t.indices[0] is mps.bonds[i]
    #        assert t.indices[1] is mps.indices[i]
    #        assert t.indices[2] is mps.bonds[i + 1]
#
    #    for i in range(L - 1):
    #        assert mps.tensors[i].indices[2] is mps.tensors[i + 1].indices[0]
#
    #def test_allocate_false_creates_structure_only(self):
    #    mps = MPS(L=3, physical_dims=2, bond_policy=[1, 2, 2, 1], allocate=False)
    #    assert mps.tensors[0].data is None
    #    assert mps.tensors[1].data is None
    #    assert mps.tensors[2].data is None
#
    #def test_allocate_true_allocates_zeros(self):
    #    mps = MPS(L=3, physical_dims=2, bond_policy=[1, 2, 2, 1], allocate=True)
    #    assert mps.tensors[0].data is not None
    #    assert mps.tensors[1].data is not None
    #    assert mps.tensors[2].data is not None
    #    assert np.all(mps.tensors[0].data == 0)
#
    #def test_default_bond_policy_heterogeneous_phys_dims(self):
    #    # physical dims: [2,3,2,2]
    #    # bond dims expected: [1, 2, 4, 2, 1]
    #    mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy="default", allocate=False)
    #    assert mps.bond_dims == [1, 2, 4, 2, 1]
#
    #def test_default_bond_policy_with_truncation_cap(self):
    #    trunc = SimpleNamespace(chi_max=3)
    #    mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy="default", truncation=trunc, allocate=False)
    #    # [1, 2, min(4,3)=3, 2, 1]
    #    assert mps.bond_dims == [1, 2, 3, 2, 1]
#
    #def test_uniform_bond_policy_requires_chi_max(self):
    #    with pytest.raises(ValueError, match="chi_max"):
    #        _ = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=None, allocate=False)
#
    #    with pytest.raises(ValueError, match="chi_max"):
    #        _ = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=SimpleNamespace(chi_max=None), allocate=False)
#
    #def test_uniform_bond_policy_uses_truncation_chi_max(self):
    #    trunc = SimpleNamespace(chi_max=7)
    #    mps = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=trunc, allocate=False)
    #    assert mps.bond_dims == [1, 7, 7, 7, 7, 1]
#
    #def test_explicit_bond_policy_wrong_length_raises(self):
    #    with pytest.raises(AssertionError):
    #        _ = MPS(L=4, physical_dims=2, bond_policy=[1, 2, 1], allocate=False)


# class TestMPSFactories:
#     def test_from_product_state_qubits_dense_matches(self):
#         state = [0, 1, 0, 1, 1]  # realistic computational basis bitstring
#         mps = MPS.from_product_state(state_indices=state, physical_dims=2, name="psi")
# 
#         dense_expected = _kron_list([_basis_vec(2, s) for s in state])
#         dense = mps.to_dense()
#         np.testing.assert_allclose(dense, dense_expected, atol=0, rtol=0)
# 
#         assert abs(mps.norm() - 1.0) < 1e-12
# 
#     def test_from_product_state_invalid_local_index_raises(self):
#         with pytest.raises(ValueError, match="Invalid local state index"):
#             _ = MPS.from_product_state(state_indices=[0, 2, 1], physical_dims=2)
# 
#     def test_from_local_states_dense_matches_and_norm(self):
#         rng = np.random.default_rng(123)
#         # realistic local states: complex amplitudes, not necessarily normalized
#         v0 = rng.normal(size=2) + 1j * rng.normal(size=2)
#         v1 = rng.normal(size=2) + 1j * rng.normal(size=2)
#         v2 = rng.normal(size=2) + 1j * rng.normal(size=2)
# 
#         mps = MPS.from_local_states([v0, v1, v2], name="prod_local")
#         dense_expected = _kron_list([v0, v1, v2])
#         dense = mps.to_dense()
#         np.testing.assert_allclose(dense, dense_expected, atol=1e-12, rtol=1e-12)
# 
#         # Norm of product state equals product of local norms
#         expected_norm = float(np.linalg.norm(v0) * np.linalg.norm(v1) * np.linalg.norm(v2))
#         assert abs(mps.norm() - expected_norm) < 1e-10
# 
#     def test_from_local_states_empty_raises(self):
#         with pytest.raises(ValueError, match="non-empty"):
#             _ = MPS.from_local_states([])
# 
#     def test_from_tensors_roundtrip(self):
#         mps0 = MPS.from_product_state([1, 0, 1, 0], physical_dims=2, name="psi0")
#         mps1 = MPS.from_tensors(mps0.tensors, name="psi1")
# 
#         assert len(mps1) == len(mps0)
#         assert mps1.physical_dims == mps0.physical_dims
#         assert mps1.bond_dims == mps0.bond_dims
# 
#         # Dense should match exactly for this deterministic state
#         np.testing.assert_allclose(mps1.to_dense(), mps0.to_dense(), atol=0, rtol=0)
# 
# 
# class TestMPSCoreMethods:
#     def test_norm_raises_for_structure_only(self):
#         mps = MPS(L=3, physical_dims=2, bond_policy=[1, 1, 1, 1], allocate=False)
#         with pytest.raises(ValueError, match="data=None"):
#             _ = mps.norm()
# 
#     def test_to_dense_raises_for_structure_only(self):
#         mps = MPS(L=2, physical_dims=2, bond_policy=[1, 1, 1], allocate=False)
#         with pytest.raises(ValueError, match="data=None"):
#             _ = mps.to_dense()
# 
#     def test_normalize_makes_unit_norm(self):
#         rng = np.random.default_rng(7)
#         # Start with a random product state (local states), intentionally unnormalized
#         v0 = rng.normal(size=2) + 1j * rng.normal(size=2)
#         v1 = rng.normal(size=2) + 1j * rng.normal(size=2)
#         v2 = rng.normal(size=2) + 1j * rng.normal(size=2)
# 
#         mps = MPS.from_local_states([v0, v1, v2], name="psi")
#         n0 = mps.norm()
#         assert n0 > 0
# 
#         # Capture first tensor copy to ensure scaling only changes its data (as implemented)
#         t0_before = mps.tensors[0].data.copy()
#         mps.normalize()
#         n1 = mps.norm()
#         assert abs(n1 - 1.0) < 1e-10
# 
#         # First tensor should be scaled by 1/n0
#         np.testing.assert_allclose(mps.tensors[0].data, t0_before * (1.0 / n0), atol=1e-10, rtol=1e-10)
# 
#     def test_copy_is_deep_for_data(self):
#         mps = MPS.from_product_state([0, 1, 1, 0], physical_dims=2, name="psi")
#         mps2 = mps.copy()
# 
#         np.testing.assert_allclose(mps2.to_dense(), mps.to_dense(), atol=0, rtol=0)
# 
#         # Mutate copy and ensure original unchanged
#         mps2.tensors[0].data[0, 0, 0] += 0.5
#         assert not np.allclose(mps2.to_dense(), mps.to_dense())
# 
#     def test_bond_connectivity_preserved_after_copy(self):
#         mps = MPS.from_product_state([0, 1, 0, 1, 0], physical_dims=2, name="psi")
#         mps2 = mps.copy()
# 
#         # In the copied object, internal bonds must still connect adjacent tensors
#         for i in range(len(mps2) - 1):
#             assert mps2.tensors[i].indices[2] is mps2.tensors[i + 1].indices[0]
