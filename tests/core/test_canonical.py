"""Tests for canonical form transformations in canonical.py.

Coverage:
  left_canonicalize  -- orthonormality of all sites except last; state preserved
  right_canonicalize -- orthonormality of all sites except first; state preserved
  mixed_canonicalize -- left/right regions orthonormal; center free; state preserved
  is_left_orthonormal / is_right_orthonormal helpers

Fixtures used:
  - product states (|0101> etc.) via MPS.from_product_state
  - random statevectors via MPS.from_statevector
  - Bell and GHZ-like states built as statevectors
"""

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.canonical import (
    left_canonicalize,
    right_canonicalize,
    mixed_canonicalize,
    is_left_orthonormal,
    is_right_orthonormal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bell_state() -> np.ndarray:
    """Return the Bell state |00> + |11> / sqrt(2) as a dense statevector."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2)  # |00>
    psi[3] = 1.0 / np.sqrt(2)  # |11>
    return psi


def _ghz_state(L: int) -> np.ndarray:
    """Return the GHZ state (|0...0> + |1...1>) / sqrt(2) for L qubits."""
    d = 2**L
    psi = np.zeros(d, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2)   # |00...0>
    psi[-1] = 1.0 / np.sqrt(2)  # |11...1>
    return psi


def _random_statevector(L: int, d: int = 2, seed: int = 42) -> np.ndarray:
    """Return a normalized random statevector for L sites with local dim d."""
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d**L) + 1j * rng.standard_normal(d**L)
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

class TestOrthonormalityHelpers:
    def test_identity_columns_are_left_orthonormal(self):
        # A trivial rank-1 tensor: product state |0>
        A = np.zeros((1, 2, 1), dtype=np.complex128)
        A[0, 0, 0] = 1.0
        assert is_left_orthonormal(A)

    def test_identity_rows_are_right_orthonormal(self):
        A = np.zeros((1, 2, 1), dtype=np.complex128)
        A[0, 0, 0] = 1.0
        assert is_right_orthonormal(A)

    def test_unnormalized_tensor_not_left_orthonormal(self):
        A = np.ones((1, 2, 1), dtype=np.complex128)  # both components = 1 -> not isometry
        assert not is_left_orthonormal(A)

    def test_unnormalized_tensor_not_right_orthonormal(self):
        A = np.ones((1, 2, 1), dtype=np.complex128)
        assert not is_right_orthonormal(A)


# ---------------------------------------------------------------------------
# left_canonicalize tests
# ---------------------------------------------------------------------------

class TestLeftCanonicalize:
    @pytest.mark.parametrize("state_indices", [
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ])
    def test_preserves_product_state(self, state_indices):
        """State vector must be unchanged after left-canonicalization."""
        mps = MPS.from_product_state(state_indices, physical_dims=2)
        psi_before = mps.to_dense()

        mps_lc = left_canonicalize(mps)
        psi_after = mps_lc.to_dense()

        assert np.allclose(psi_before, psi_after, atol=1e-12), (
            f"State changed after left_canonicalize: max diff = "
            f"{np.max(np.abs(psi_before - psi_after)):.2e}"
        )

    def test_left_orthonormality_product_state(self):
        """All sites except the last must be left-orthonormal."""
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        mps_lc = left_canonicalize(mps)

        for i, t in enumerate(mps_lc.tensors[:-1]):
            assert is_left_orthonormal(t.data), (
                f"Site {i} is not left-orthonormal after left_canonicalize"
            )

    def test_preserves_bell_state(self):
        psi = _bell_state()
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_lc = left_canonicalize(mps)
        psi_after = mps_lc.to_dense()

        # Statevectors may differ by a global phase; check magnitudes
        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    def test_left_orthonormality_bell_state(self):
        psi = _bell_state()
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_lc = left_canonicalize(mps)

        for i, t in enumerate(mps_lc.tensors[:-1]):
            assert is_left_orthonormal(t.data), f"Site {i} not left-orthonormal (Bell)"

    @pytest.mark.parametrize("L", [3, 4, 5])
    def test_preserves_ghz_state(self, L):
        psi = _ghz_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_lc = left_canonicalize(mps)
        psi_after = mps_lc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12), (
            f"GHZ L={L}: max diff = {np.max(np.abs(np.abs(psi_before) - np.abs(psi_after))):.2e}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_preserves_random_state(self, seed):
        psi = _random_statevector(L=4, d=2, seed=seed)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_lc = left_canonicalize(mps)
        psi_after = mps_lc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_left_orthonormality_random_state(self, seed):
        psi = _random_statevector(L=4, d=2, seed=seed)
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_lc = left_canonicalize(mps)

        for i, t in enumerate(mps_lc.tensors[:-1]):
            assert is_left_orthonormal(t.data), (
                f"Seed {seed}: site {i} not left-orthonormal"
            )

    def test_does_not_modify_input(self):
        """left_canonicalize must not modify the original MPS."""
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        psi_before = mps.to_dense().copy()
        _ = left_canonicalize(mps)
        assert np.allclose(mps.to_dense(), psi_before)


# ---------------------------------------------------------------------------
# right_canonicalize tests
# ---------------------------------------------------------------------------

class TestRightCanonicalize:
    @pytest.mark.parametrize("state_indices", [
        [0, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ])
    def test_preserves_product_state(self, state_indices):
        mps = MPS.from_product_state(state_indices, physical_dims=2)
        psi_before = mps.to_dense()

        mps_rc = right_canonicalize(mps)
        psi_after = mps_rc.to_dense()

        assert np.allclose(psi_before, psi_after, atol=1e-12)

    def test_right_orthonormality_product_state(self):
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        mps_rc = right_canonicalize(mps)

        for i, t in enumerate(mps_rc.tensors[1:], start=1):
            assert is_right_orthonormal(t.data), (
                f"Site {i} is not right-orthonormal after right_canonicalize"
            )

    def test_preserves_bell_state(self):
        psi = _bell_state()
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_rc = right_canonicalize(mps)
        psi_after = mps_rc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    def test_right_orthonormality_bell_state(self):
        psi = _bell_state()
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_rc = right_canonicalize(mps)

        for i, t in enumerate(mps_rc.tensors[1:], start=1):
            assert is_right_orthonormal(t.data), f"Site {i} not right-orthonormal (Bell)"

    @pytest.mark.parametrize("L", [3, 4, 5])
    def test_preserves_ghz_state(self, L):
        psi = _ghz_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_rc = right_canonicalize(mps)
        psi_after = mps_rc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_preserves_random_state(self, seed):
        psi = _random_statevector(L=4, d=2, seed=seed)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_rc = right_canonicalize(mps)
        psi_after = mps_rc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    @pytest.mark.parametrize("seed", [0, 1, 7])
    def test_right_orthonormality_random_state(self, seed):
        psi = _random_statevector(L=4, d=2, seed=seed)
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_rc = right_canonicalize(mps)

        for i, t in enumerate(mps_rc.tensors[1:], start=1):
            assert is_right_orthonormal(t.data), (
                f"Seed {seed}: site {i} not right-orthonormal"
            )

    def test_does_not_modify_input(self):
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        psi_before = mps.to_dense().copy()
        _ = right_canonicalize(mps)
        assert np.allclose(mps.to_dense(), psi_before)


# ---------------------------------------------------------------------------
# mixed_canonicalize tests
# ---------------------------------------------------------------------------

class TestMixedCanonicalize:
    @pytest.mark.parametrize("center", [0, 1, 2, 3])
    def test_preserves_product_state(self, center):
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        psi_before = mps.to_dense()

        mps_mc = mixed_canonicalize(mps, center=center)
        psi_after = mps_mc.to_dense()

        assert np.allclose(psi_before, psi_after, atol=1e-12), (
            f"center={center}: state changed, max diff = "
            f"{np.max(np.abs(psi_before - psi_after)):.2e}"
        )

    @pytest.mark.parametrize("center", [0, 1, 2, 3])
    def test_left_region_is_left_orthonormal(self, center):
        psi = _random_statevector(L=4, seed=42)
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_mc = mixed_canonicalize(mps, center=center)

        for i in range(center):
            assert is_left_orthonormal(mps_mc.tensors[i].data), (
                f"center={center}: site {i} (left of center) not left-orthonormal"
            )

    @pytest.mark.parametrize("center", [0, 1, 2, 3])
    def test_right_region_is_right_orthonormal(self, center):
        psi = _random_statevector(L=4, seed=42)
        mps = MPS.from_statevector(psi, physical_dims=2)
        mps_mc = mixed_canonicalize(mps, center=center)

        for i in range(center + 1, mps_mc.L):
            assert is_right_orthonormal(mps_mc.tensors[i].data), (
                f"center={center}: site {i} (right of center) not right-orthonormal"
            )

    @pytest.mark.parametrize("center", [0, 1, 2, 3])
    def test_preserves_bell_state(self, center):
        # L=2: valid centers are 0 and 1
        if center > 1:
            pytest.skip("center out of range for Bell (L=2)")
        psi = _bell_state()
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_mc = mixed_canonicalize(mps, center=center)
        psi_after = mps_mc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    @pytest.mark.parametrize("L,center", [(3, 1), (4, 2), (5, 2), (5, 0), (5, 4)])
    def test_preserves_ghz_state(self, L, center):
        psi = _ghz_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_mc = mixed_canonicalize(mps, center=center)
        psi_after = mps_mc.to_dense()

        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

    def test_invalid_center_raises(self):
        mps = MPS.from_product_state([0, 1, 0], physical_dims=2)
        with pytest.raises(ValueError, match="center"):
            mixed_canonicalize(mps, center=-1)
        with pytest.raises(ValueError, match="center"):
            mixed_canonicalize(mps, center=3)

    def test_does_not_modify_input(self):
        mps = MPS.from_product_state([0, 1, 0, 1], physical_dims=2)
        psi_before = mps.to_dense().copy()
        _ = mixed_canonicalize(mps, center=2)
        assert np.allclose(mps.to_dense(), psi_before)

    @pytest.mark.parametrize("seed", [0, 3, 9])
    def test_random_state_full_orthonormality(self, seed):
        """Full sanity: left + right regions orthonormal, state preserved."""
        L = 5
        center = 2
        psi = _random_statevector(L=L, seed=seed)
        mps = MPS.from_statevector(psi, physical_dims=2)
        psi_before = mps.to_dense()

        mps_mc = mixed_canonicalize(mps, center=center)
        psi_after = mps_mc.to_dense()

        # State preserved
        assert np.allclose(np.abs(psi_before), np.abs(psi_after), atol=1e-12)

        # Left region
        for i in range(center):
            assert is_left_orthonormal(mps_mc.tensors[i].data)

        # Right region
        for i in range(center + 1, L):
            assert is_right_orthonormal(mps_mc.tensors[i].data)
