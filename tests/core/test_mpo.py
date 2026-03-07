import numpy as np
import pytest

from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.mps import MPS


def kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def test_mpo_constructor_basic_properties():
    L = 4
    d = 2
    mpo = MPO(L=L, d=d, bond_policy=3)

    assert len(mpo) == L
    assert mpo.L == L
    assert mpo.d == d
    assert mpo.bond_dims == [1, 3, 3, 3, 1]
    assert mpo.physical_dims == [d] * L
    assert len(mpo.shape) == L

    for i, tensor in enumerate(mpo.tensors):
        assert tensor.shape == (mpo.bond_dims[i], d, d, mpo.bond_dims[i + 1])


def test_mpo_getitem_returns_site_tensor():
    mpo = MPO(L=3, d=2, bond_policy=2)
    assert mpo[0] is mpo.tensors[0]
    assert mpo[1] is mpo.tensors[1]
    assert mpo[2] is mpo.tensors[2]


def test_identity_mpo_shapes_and_dense():
    L = 3
    d = 2
    mpo = MPO.identity_mpo(L=L, d=d)

    assert len(mpo) == L
    assert mpo.bond_dims == [1] * (L + 1)
    assert mpo.physical_dims == [d] * L

    for tensor in mpo.tensors:
        assert tensor.shape == (1, d, d, 1)

    dense = mpo.to_dense()
    expected = np.eye(d**L, dtype=np.complex128)
    assert dense.shape == (d**L, d**L)
    assert np.allclose(dense, expected)


def test_copy_is_deep_copy():
    mpo = MPO.identity_mpo(L=2, d=2)
    mpo_copy = mpo.copy()

    assert mpo_copy is not mpo
    assert mpo_copy.L == mpo.L
    assert mpo_copy.d == mpo.d
    assert mpo_copy.bond_dims == mpo.bond_dims

    for t_orig, t_copy in zip(mpo.tensors, mpo_copy.tensors):
        assert t_copy is not t_orig
        assert np.allclose(t_copy.data, t_orig.data)

    mpo_copy.tensors[0].data[0, 0, 0, 0] = 7.0
    assert not np.allclose(mpo_copy.tensors[0].data, mpo.tensors[0].data)


def test_initialize_random_preserves_shapes_and_changes_data():
    mpo = MPO(L=3, d=2, bond_policy=2)
    before = [t.data.copy() for t in mpo.tensors]

    mpo.initialize_random()

    for t, old in zip(mpo.tensors, before):
        assert t.shape == old.shape
        assert t.data.dtype == np.complex128
        assert not np.allclose(t.data, old)


def test_initialize_single_site_operator_dense_matches_expected():
    L = 3
    d = 2
    site = 1

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    mpo = MPO(L=L, d=d, bond_policy=1)
    mpo.initialize_single_site_operator(X, site=site)

    dense = mpo.to_dense()

    I = np.eye(d, dtype=np.complex128)
    expected = kron_all([I, X, I])

    assert dense.shape == expected.shape
    assert np.allclose(dense, expected)


def test_initialize_single_site_operator_invalid_shape_raises():
    mpo = MPO(L=3, d=2, bond_policy=1)
    bad_op = np.eye(3, dtype=np.complex128)

    with pytest.raises(ValueError, match="Operator shape"):
        mpo.initialize_single_site_operator(bad_op, site=1)


def test_initialize_single_site_operator_invalid_site_raises():
    mpo = MPO(L=3, d=2, bond_policy=1)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="out of range"):
        mpo.initialize_single_site_operator(X, site=3)


def test_to_dense_for_single_site_operator_L1():
    d = 2
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    mpo = MPO(L=1, d=d, bond_policy=1)
    mpo.initialize_single_site_operator(Z, site=0)

    dense = mpo.to_dense()
    assert dense.shape == (d, d)
    assert np.allclose(dense, Z)


def test_apply_identity_mpo_returns_same_state_dense():
    state = [0, 1, 0]
    d = 2

    mps = MPS.from_product_state(state, physical_dims=d)
    mpo = MPO.identity_mpo(L=len(state), d=d)

    new_mps = mpo.apply(mps)

    assert np.allclose(new_mps.to_dense(), mps.to_dense())


def test_apply_matches_dense_reference_for_single_site_operator():
    state = [0, 1, 1]
    d = 2
    site = 0

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    mps = MPS.from_product_state(state, physical_dims=d)

    mpo = MPO(L=len(state), d=d, bond_policy=1)
    mpo.initialize_single_site_operator(X, site=site)

    mps_out = mpo.apply(mps)

    psi = mps.to_dense()
    O = mpo.to_dense()
    expected = O @ psi

    assert np.allclose(mps_out.to_dense(), expected)


def test_apply_length_mismatch_raises():
    mpo = MPO.identity_mpo(L=3, d=2)
    mps = MPS.from_product_state([0, 1], physical_dims=2)

    with pytest.raises(ValueError, match="Length mismatch"):
        mpo.apply(mps)


def test_apply_physical_dimension_mismatch_raises():
    mpo = MPO.identity_mpo(L=3, d=3)
    mps = MPS.from_product_state([0, 1, 0], physical_dims=2)

    with pytest.raises(ValueError, match="Physical dimension mismatch"):
        mpo.apply(mps)


def test_repr_contains_core_information():
    mpo = MPO(L=2, d=2, bond_policy=2)
    rep = repr(mpo)

    assert "MPO" in rep
    assert "L=2" in rep
    assert "d=2" in rep
    assert "shapes=" in rep


def test_custom_bond_policy_list_is_respected():
    bond_dims = [1, 4, 3, 1]
    mpo = MPO(L=3, d=2, bond_policy=bond_dims)

    assert mpo.bond_dims == bond_dims
    assert mpo.tensors[0].shape == (1, 2, 2, 4)
    assert mpo.tensors[1].shape == (4, 2, 2, 3)
    assert mpo.tensors[2].shape == (3, 2, 2, 1)


def test_invalid_bond_policy_length_raises():
    with pytest.raises(ValueError, match=r"bond_policy list must have length"):
        MPO(L=3, d=2, bond_policy=[1, 2, 1])
 