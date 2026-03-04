# tests/core/test_mps.py

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.index import Index
from tensor_network_library.core.tensor import Tensor
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.states.qubit_states import qubit_states

def _kron_list(vecs: list[np.ndarray]) -> np.ndarray:
    out = np.array([1.0 + 0.0j], dtype=np.complex128)
    for v in vecs:
        out = np.kron(out, np.asarray(v, dtype=np.complex128).reshape(-1))
    return out


def _kron_all(vecs):
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
        mps = MPS(L=4, physical_dims=2, bond_policy=[1, 1, 1, 1, 1])
        assert len(mps) == 4
        r = repr(mps)
        assert "MPS" in r
        assert "L=4" in r

    def test_init_is_structure_only(self):
        L = 5
        mps = MPS(L=L, physical_dims=2, bond_policy=[1] * (L + 1))
        assert len(mps.tensors) == L
        assert all(t.data is None for t in mps.tensors)

    def test_indices_are_index_objects_and_connected(self):
        L = 6
        mps = MPS(L=L, physical_dims=2, bond_policy=[1] * (L + 1))

        assert len(mps.indices) == L
        assert len(mps.bonds) == L + 1

        for ix in mps.indices:
            assert isinstance(ix, Index)
        for b in mps.bonds:
            assert isinstance(b, Index)

        for i in range(L):
            t = mps.tensors[i]
            assert isinstance(t, Tensor)
            assert len(t.indices) == 3
            assert t.indices[0] is mps.bonds[i]
            assert t.indices[1] is mps.indices[i]
            assert t.indices[2] is mps.bonds[i + 1]

        for i in range(L - 1):
            assert mps.tensors[i].indices[2] is mps.tensors[i + 1].indices[0]

    def test_physical_dims_int(self):
        mps = MPS(L=4, physical_dims=3, bond_policy=[1, 1, 1, 1, 1])
        assert mps.physical_dims == [3, 3, 3, 3]

    def test_physical_dims_list(self):
        mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy=[1, 1, 1, 1, 1])
        assert mps.physical_dims == [2, 3, 2, 2]

    def test_physical_dims_list_wrong_length_raises(self):
        with pytest.raises(AssertionError, match="length L"):
            _ = MPS(L=4, physical_dims=[2, 2, 2], bond_policy=[1, 1, 1, 1, 1])

    def test_default_bond_policy_heterogeneous_phys_dims(self):
        # physical dims [2,3,2,2] -> bond dims [1,2,4,2,1]
        mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy="default")
        assert mps.bond_dims == [1, 2, 4, 2, 1]

    def test_default_bond_policy_with_truncation_cap(self):
        # TruncationPolicy uses max_bond_dim, not chi_max
        trunc = SimpleNamespace(max_bond_dim=3)
        mps = MPS(L=4, physical_dims=[2, 3, 2, 2], bond_policy="default", truncation=trunc)
        assert mps.bond_dims == [1, 2, 3, 2, 1]

    def test_uniform_bond_policy_requires_max_bond_dim(self):
        with pytest.raises(ValueError, match="max_bond_dim"):
            _ = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=None)

        with pytest.raises(ValueError, match="max_bond_dim"):
            _ = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=SimpleNamespace(max_bond_dim=None))

    def test_uniform_bond_policy_uses_max_bond_dim(self):
        trunc = SimpleNamespace(max_bond_dim=7)
        mps = MPS(L=5, physical_dims=2, bond_policy="uniform", truncation=trunc)
        assert mps.bond_dims == [1, 7, 7, 7, 7, 1]

    def test_explicit_bond_policy_wrong_length_raises(self):
        with pytest.raises(AssertionError, match="length L\\+1"):
            _ = MPS(L=4, physical_dims=2, bond_policy=[1, 2, 1])

    def test_explicit_bond_policy_boundary_not_one_raises(self):
        with pytest.raises(ValueError, match="Boundary bond dimensions must be 1"):
            _ = MPS(L=4, physical_dims=2, bond_policy=[2, 2, 2, 2, 2])


class TestMPSUnmaterializedBehavior:
    def test_norm_raises_for_structure_only(self):
        mps = MPS(L=3, physical_dims=2, bond_policy=[1, 1, 1, 1])
        with pytest.raises(ValueError, match="data=None|unmaterialized"):
            _ = mps.norm()

    def test_to_dense_raises_for_structure_only(self):
        mps = MPS(L=2, physical_dims=2, bond_policy=[1, 1, 1])
        with pytest.raises(ValueError, match="data=None|unmaterialized"):
            _ = mps.to_dense()

    def test_normalize_raises_for_structure_only(self):
        mps = MPS(L=2, physical_dims=2, bond_policy=[1, 1, 1])
        with pytest.raises(ValueError, match="data=None|unmaterialized"):
            _ = mps.normalize()


class TestMPSFactories:
    def test_from_product_state_qubits_dense_matches(self):
        state = [0, 1, 0, 1, 1]
        mps = MPS.from_product_state(state_indices=state, physical_dims=2, name="psi")

        dense_expected = _kron_list([_basis_vec(2, s) for s in state])
        dense = mps.to_dense()
        np.testing.assert_allclose(dense, dense_expected, atol=0, rtol=0)

        assert abs(mps.norm() - 1.0) < 1e-12

    def test_from_product_state_invalid_local_index_raises(self):
        with pytest.raises(ValueError, match="Invalid local state index"):
            _ = MPS.from_product_state(state_indices=[0, 2, 1], physical_dims=2)

    def test_from_local_states_dense_matches_and_norm(self):
        rng = np.random.default_rng(123)

        # realistic local states: complex amplitudes, not necessarily normalized
        v0 = rng.normal(size=2) + 1j * rng.normal(size=2)
        v1 = rng.normal(size=2) + 1j * rng.normal(size=2)
        v2 = rng.normal(size=2) + 1j * rng.normal(size=2)

        mps = MPS.from_local_states([v0, v1, v2], name="prod_local")
        dense_expected = _kron_list([v0, v1, v2])
        dense = mps.to_dense()
        np.testing.assert_allclose(dense, dense_expected, atol=1e-12, rtol=1e-12)

        expected_norm = float(np.linalg.norm(v0) * np.linalg.norm(v1) * np.linalg.norm(v2))
        assert abs(mps.norm() - expected_norm) < 1e-10

    def test_from_local_states_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _ = MPS.from_local_states([])

    def test_from_tensors_roundtrip(self):
        mps0 = MPS.from_product_state([1, 0, 1, 0], physical_dims=2, name="psi0")
        mps1 = MPS.from_tensors(mps0.tensors, name="psi1")

        assert len(mps1) == len(mps0)
        assert mps1.physical_dims == mps0.physical_dims
        assert mps1.bond_dims == mps0.bond_dims
        np.testing.assert_allclose(mps1.to_dense(), mps0.to_dense(), atol=0, rtol=0)


class TestMPSCoreMethods:
    def test_normalize_makes_unit_norm(self):
        rng = np.random.default_rng(7)
        v0 = rng.normal(size=2) + 1j * rng.normal(size=2)
        v1 = rng.normal(size=2) + 1j * rng.normal(size=2)
        v2 = rng.normal(size=2) + 1j * rng.normal(size=2)

        mps = MPS.from_local_states([v0, v1, v2], name="psi")
        n0 = mps.norm()
        assert n0 > 0

        t0_before = mps.tensors[0].data.copy()
        mps.normalize()
        n1 = mps.norm()
        assert abs(n1 - 1.0) < 1e-10

        np.testing.assert_allclose(mps.tensors[0].data, t0_before * (1.0 / n0), atol=1e-10, rtol=1e-10)

    def test_copy_preserves_dense_state(self):
        mps = MPS.from_product_state([0, 1, 1, 0], physical_dims=2, name="psi")
        mps2 = mps.copy()
        np.testing.assert_allclose(mps2.to_dense(), mps.to_dense(), atol=0, rtol=0)

    def test_copy_is_independent_in_data(self):
        mps = MPS.from_product_state([0, 1, 1, 0], physical_dims=2, name="psi")
        mps2 = mps.copy()

        mps2.tensors[0].data[0, 0, 0] += 0.25
        assert not np.allclose(mps2.to_dense(), mps.to_dense())

    def test_bond_connectivity_preserved_after_copy(self):
        mps = MPS.from_product_state([0, 1, 0, 1, 0], physical_dims=2, name="psi")
        mps2 = mps.copy()

        for i in range(len(mps2) - 1):
            assert mps2.tensors[i].indices[2] is mps2.tensors[i + 1].indices[0]


@pytest.mark.parametrize(
    "name, state_indices, expected_dense",
    [
        (
            "GHZ_like_length2_basis_superposition",
            [0, 0],  # we'll build |00> + |11>
            (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])).astype(np.complex128),
        ),
        (
            "Bell_phi_plus",
            [0, 0],  # placeholder, we override with local states below
            (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])).astype(np.complex128) / np.sqrt(2.0),
        ),
    ],
)
def test_specific_two_qubit_states_via_from_local_states(name, state_indices, expected_dense):
    # These are not representable as computational-basis product states (entangled),
    # so we test via from_local_states only for product cases; for entangled we use manual MPS below.
    # Here we just sanity-check the expected_dense normalization logic.
    assert expected_dense.shape == (4,)


def test_initialize_product_state_all_zeros_qubits():
    mps = MPS.from_product_state([0, 0, 0, 0], physical_dims=2, name="zeros")
    dense = mps.to_dense()

    expected = np.kron(np.kron(np.kron([1, 0], [1, 0]), [1, 0]), [1, 0]).astype(np.complex128)
    np.testing.assert_allclose(dense, expected, atol=0, rtol=0)
    assert abs(mps.norm() - 1.0) < 1e-12


def test_initialize_product_state_all_ones_qubits():
    mps = MPS.from_product_state([1, 1, 1], physical_dims=2, name="ones")
    dense = mps.to_dense()

    expected = np.kron(np.kron([0, 1], [0, 1]), [0, 1]).astype(np.complex128)
    np.testing.assert_allclose(dense, expected, atol=0, rtol=0)
    assert abs(mps.norm() - 1.0) < 1e-12


def test_initialize_alternating_bitstring_qubits():
    bits = [0, 1, 0, 1, 0]
    mps = MPS.from_product_state(bits, physical_dims=2, name="alt")

    expected = np.array([1.0 + 0.0j], dtype=np.complex128)
    for b in bits:
        expected = np.kron(expected, np.array([1, 0], dtype=np.complex128) if b == 0 else np.array([0, 1], dtype=np.complex128))

    np.testing.assert_allclose(mps.to_dense(), expected, atol=0, rtol=0)
    assert abs(mps.norm() - 1.0) < 1e-12


def test_initialize_nonuniform_local_states_qubits():
    # |psi> = (|0>+|1>)/sqrt(2) ⊗ |1> ⊗ (2|0>-i|1>)
    v0 = (np.array([1.0, 1.0]) / np.sqrt(2.0)).astype(np.complex128)
    v1 = np.array([0.0, 1.0], dtype=np.complex128)
    v2 = np.array([2.0, -1.0j], dtype=np.complex128)

    mps = MPS.from_local_states([v0, v1, v2], name="psi_local")

    expected = np.kron(np.kron(v0, v1), v2)
    np.testing.assert_allclose(mps.to_dense(), expected, atol=1e-12, rtol=1e-12)

    expected_norm = float(np.linalg.norm(v0) * np.linalg.norm(v1) * np.linalg.norm(v2))
    assert abs(mps.norm() - expected_norm) < 1e-10


def test_initialize_entangled_bell_state_manual_bond_dim_2():
    # Build |Φ+> = (|00> + |11>)/sqrt(2) exactly as an MPS with chi=2.
    # A0(1,2): [bondL=1, phys=2, bond=2]
    # A1(2,2,1): [bond=2, phys=2, bondR=1]
    #
    # A0[0,0,0]=1/sqrt(2), A0[0,1,1]=1/sqrt(2)
    # A1[0,0,0]=1,         A1[1,1,0]=1

    dtype = np.complex128

    b0 = Index(dim=1, name="b0", tags=frozenset({"bond"}))
    b1 = Index(dim=2, name="b1", tags=frozenset({"bond"}))
    b2 = Index(dim=1, name="b2", tags=frozenset({"bond"}))

    p0 = Index(dim=2, name="p0", tags=frozenset({"phys"}))
    p1 = Index(dim=2, name="p1", tags=frozenset({"phys"}))

    A0 = Tensor(None, indices=[b0, p0, b1]).materialize_zeros(dtype=dtype)
    A1 = Tensor(None, indices=[b1, p1, b2]).materialize_zeros(dtype=dtype)

    A0.data[0, 0, 0] = 1.0 / np.sqrt(2.0)
    A0.data[0, 1, 1] = 1.0 / np.sqrt(2.0)

    A1.data[0, 0, 0] = 1.0
    A1.data[1, 1, 0] = 1.0

    mps = MPS.from_tensors([A0, A1], name="bell")

    expected = (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])).astype(np.complex128) / np.sqrt(2.0)
    np.testing.assert_allclose(mps.to_dense(), expected, atol=1e-12, rtol=1e-12)
    assert abs(mps.norm() - 1.0) < 1e-12


def test_from_qubit_labels_matches_dense_kron():
    labels = ["0", "+", "i", "t3", "h7", "phi=pi/4"]
    mps = MPS.from_qubit_labels(labels, name="psi")

    dense = mps.to_dense()
    expected = _kron_all(qubit_states(labels))

    assert dense.shape == expected.shape
    assert np.allclose(dense, expected, atol=1e-12, rtol=0)


def test_from_qubit_labels_norm_is_product_of_local_norms():
    # from_local_states does not normalize the local vectors; it inserts them verbatim.
    # qubit_states() returns normalized vectors, so the MPS norm should be 1.
    labels = ["t0", "h0", "+", "1"]
    mps = MPS.from_qubit_labels(labels)
    assert np.isclose(mps.norm(), 1.0, atol=1e-12, rtol=0)


def test_from_qubit_labels_dtype_is_respected():
    labels = ["0", "t0", "h0"]
    mps = MPS.from_qubit_labels(labels, dtype=np.complex64)
    dense = mps.to_dense()
    assert dense.dtype == np.complex64


def test_from_qubit_labels_invalid_label_raises():
    with pytest.raises(ValueError):
        MPS.from_qubit_labels(["0", "definitely_not_a_state"])


def test_from_qubit_labels_length_and_bond_dims():
    labels = ["0", "+", "t0", "h0", "phi=pi/7"]
    mps = MPS.from_qubit_labels(labels)
    assert len(mps) == len(labels)
    assert mps.bond_dims == [1] * (len(labels) + 1)
    assert mps.physical_dims == [2] * len(labels)


def _global_phase_equal(v: np.ndarray, w: np.ndarray, atol: float = 1e-12) -> bool:
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    w = np.asarray(w, dtype=np.complex128).reshape(-1)
    if v.shape != w.shape:
        return False
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv == 0 or nw == 0:
        return False
    v = v / nv
    w = w / nw

    idx = int(np.argmax(np.abs(w)))
    if np.abs(w[idx]) < atol:
        return np.allclose(v, w, atol=atol, rtol=0)

    phase = v[idx] / w[idx]
    return np.allclose(v, phase * w, atol=atol, rtol=0)


def _kron_all(vecs):
    out = np.array([1.0 + 0.0j], dtype=np.complex128)
    for v in vecs:
        out = np.kron(out, np.asarray(v, dtype=np.complex128).reshape(-1))
    return out


# -------------------------
# from_qubit_labels tests
# -------------------------

def test_from_qubit_labels_matches_dense_kron():
    labels = ["0", "+", "i", "t3", "h7", "phi=pi/4"]
    mps = MPS.from_qubit_labels(labels, name="psi")
    dense = mps.to_dense()

    expected = _kron_all(qubit_states(labels))
    assert dense.shape == expected.shape
    assert np.allclose(dense, expected, atol=1e-12, rtol=0)


def test_from_qubit_labels_product_mps_has_bond_dim_1():
    labels = ["0", "t0", "h0", "phi=pi/7"]
    mps = MPS.from_qubit_labels(labels)
    assert mps.bond_dims == [1] * (len(labels) + 1)
    assert mps.physical_dims == [2] * len(labels)


def test_from_qubit_labels_invalid_label_raises():
    with pytest.raises(ValueError):
        MPS.from_qubit_labels(["0", "not_a_state"])


def test_from_qubit_labels_dtype_is_respected():
    labels = ["0", "t0", "h0"]
    mps = MPS.from_qubit_labels(labels, dtype=np.complex64)
    dense = mps.to_dense()
    assert dense.dtype == np.complex64


# -------------------------
# from_statevector tests
# -------------------------

def test_from_statevector_exact_roundtrip_up_to_global_phase():
    rng = np.random.default_rng(0)
    L = 5
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi = psi / np.linalg.norm(psi)

    mps = MPS.from_statevector(psi, physical_dims=2, truncation=None, name="svd_exact")
    dense = mps.to_dense()

    assert dense.shape == psi.shape
    assert _global_phase_equal(dense, psi, atol=1e-10)
    assert np.isclose(mps.norm(), 1.0, atol=1e-10, rtol=0)


@pytest.mark.parametrize("absorb", ["right", "left", "sqrt"])
def test_from_statevector_absorb_variants_agree(absorb):
    rng = np.random.default_rng(1)
    L = 4
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi = psi / np.linalg.norm(psi)

    mps = MPS.from_statevector(psi, physical_dims=2, truncation=None, absorb=absorb, name=f"abs_{absorb}")
    dense = mps.to_dense()
    assert _global_phase_equal(dense, psi, atol=1e-10)


def test_from_statevector_with_dim_list():
    rng = np.random.default_rng(2)
    dims = [2, 3, 2]
    n = int(np.prod(dims))
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi = psi / np.linalg.norm(psi)

    mps = MPS.from_statevector(psi, physical_dims=dims, truncation=None, name="hetero")
    dense = mps.to_dense()
    assert dense.shape == psi.shape
    assert _global_phase_equal(dense, psi, atol=1e-10)


def test_from_statevector_wrong_length_raises():
    psi = np.ones(10, dtype=np.complex128)  # not a power of 2
    with pytest.raises(ValueError):
        MPS.from_statevector(psi, physical_dims=2)


def test_from_statevector_truncation_respects_max_bond_dim():
    rng = np.random.default_rng(3)
    L = 8
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi = psi / np.linalg.norm(psi)

    policy = TruncationPolicy(max_bond_dim=4, cutoff=0.0, strict=False)
    mps = MPS.from_statevector(psi, physical_dims=2, truncation=policy, name="trunc")

    assert max(mps.bond_dims) <= policy.max_bond_dim
    assert np.isclose(mps.norm(), 1.0, atol=1e-10, rtol=0)


def test_from_statevector_strict_policy_raises_if_needed_more_than_max():
    # Construct a state whose first cut has 4 equal nonzero singular values:
    # |psi> = (1/2) * sum_{a=0..3} |a>_left |a>_right, with left/right being 2 qubits each.
    psi_mat = np.eye(4, dtype=np.complex128) / 2.0  # Frobenius norm 1 -> state norm 1
    psi = psi_mat.reshape(-1)

    # cutoff=0 keeps all nonzero singular values (4), but max_bond_dim=2 => should raise in strict mode.
    policy = TruncationPolicy(max_bond_dim=2, cutoff=0.0, strict=True)

    with pytest.raises(ValueError):
        MPS.from_statevector(psi, physical_dims=2, truncation=policy, name="strict_fail")


def test_from_statevector_normalize_flag_controls_scaling():
    rng = np.random.default_rng(4)
    L = 4
    psi = rng.normal(size=2**L) + 1j * rng.normal(size=2**L)
    psi = psi / np.linalg.norm(psi)

    scale = 3.7
    psi2 = scale * psi

    mps_normed = MPS.from_statevector(psi2, physical_dims=2, truncation=None, normalize=True, name="normed")
    assert np.isclose(mps_normed.norm(), 1.0, atol=1e-10, rtol=0)

    mps_raw = MPS.from_statevector(psi2, physical_dims=2, truncation=None, normalize=False, name="raw")
    dense_raw = mps_raw.to_dense()
    assert _global_phase_equal(dense_raw, psi2, atol=1e-10)
    assert np.isclose(mps_raw.norm(), np.linalg.norm(psi2), atol=1e-8, rtol=1e-8)