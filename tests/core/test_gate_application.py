# tests/core/test_gate_application.py
from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.core.gate_application import apply_two_site_gate
from tensor_network_library.states.entangled_states import (
    bell_statevector,
    ghz_statevector,
)
from tensor_network_library.core.mps import MPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _product_mps(L: int, d: int = 2, dtype=np.complex128) -> MPS:
    vec = np.zeros(d**L, dtype=dtype)
    vec[0] = 1.0
    return MPS.from_statevector(vec, physical_dims=d, normalize=True, dtype=dtype)


def _norm_mps(mps: MPS) -> float:
    return float(np.linalg.norm(mps.to_dense()))


def _I() -> np.ndarray:
    return np.eye(2, dtype=np.complex128)

def _X() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)

def _H_gate() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)

def _CNOT() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )

def _SWAP() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )

def _kron(A, B):
    return np.kron(A, B)


# ---------------------------------------------------------------------------
# Norm conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gate", [_CNOT(), _SWAP()])
@pytest.mark.parametrize("site_i", [0, 1, 2])
def test_unitary_gate_preserves_norm(gate, site_i):
    L = 4
    mps = _product_mps(L)
    result, _ = apply_two_site_gate(mps, gate, site_i=site_i)
    assert np.isclose(_norm_mps(result), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Gate acts correctly on known states
# ---------------------------------------------------------------------------

def test_cnot_on_all_zero_product_state():
    L = 2
    mps = _product_mps(L)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=0)
    v = result.to_dense()
    ref = np.zeros(4, dtype=np.complex128)
    ref[0] = 1.0
    assert np.allclose(v, ref, atol=1e-12)


def test_cnot_on_x_zero_state():
    """|10> --CNOT--> |11>"""
    L = 2
    vec = np.zeros(4, dtype=np.complex128)
    vec[2] = 1.0
    mps = MPS.from_statevector(vec, physical_dims=2, normalize=True)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=0)
    v = result.to_dense()
    ref = np.zeros(4, dtype=np.complex128)
    ref[3] = 1.0
    assert np.allclose(v, ref, atol=1e-12)


def test_hadamard_cnot_creates_bell_state():
    """H⊗I followed by CNOT on |00> -> |Φ+>"""
    L = 2
    mps = _product_mps(L)
    H_I = _kron(_H_gate(), _I())
    mps, _ = apply_two_site_gate(mps, H_I, site_i=0)
    mps, _ = apply_two_site_gate(mps, _CNOT(), site_i=0)
    v = mps.to_dense()
    ref = bell_statevector(L=2, which="phi+")
    assert np.allclose(np.abs(v), np.abs(ref), atol=1e-12)


def test_swap_gate_swaps_sites():
    """|10> --SWAP--> |01>"""
    L = 2
    vec = np.zeros(4, dtype=np.complex128)
    vec[2] = 1.0
    mps = MPS.from_statevector(vec, physical_dims=2, normalize=True)
    result, _ = apply_two_site_gate(mps, _SWAP(), site_i=0)
    v = result.to_dense()
    ref = np.zeros(4, dtype=np.complex128)
    ref[1] = 1.0
    assert np.allclose(v, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Gate acts on correct sites in longer chains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [0, 1, 2])
def test_cnot_acts_only_on_target_sites(target):
    L = 4
    vec = np.zeros(2**L, dtype=np.complex128)
    idx = 1 << (L - 1 - target)
    vec[idx] = 1.0
    mps = MPS.from_statevector(vec, physical_dims=2, normalize=True)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=target)
    v = result.to_dense()
    ref = np.zeros(2**L, dtype=np.complex128)
    ref_idx = (1 << (L - 1 - target)) | (1 << (L - 2 - target))
    ref[ref_idx] = 1.0
    assert np.allclose(v, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_max_bond_limits_bond_dimension():
    L = 4
    psi = ghz_statevector(L=L)
    mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=1, max_bond=2)
    chi = result.tensors[1].data.shape[2]
    assert chi <= 2


def test_svd_cutoff_reduces_bond_dim():
    """A large svd_cutoff should discard small singular values."""
    L = 4
    # Build a state with one large and one tiny singular value at bond 1→2.
    # Mix GHZ (bond dim 2, equal SVs ~0.707) with a tiny perturbation so that
    # after CNOT the bond has one dominant SV and one negligible one.
    # Simpler: just assert that cutoff=0.0 keeps all and cutoff=1.0 keeps 1.
    psi = ghz_statevector(L=L)
    mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

    # cutoff above all singular values -> keeps only 1
    result, S = apply_two_site_gate(
        mps, _CNOT(), site_i=1, svd_cutoff=1.0
    )
    chi = result.tensors[1].data.shape[2]
    assert chi == 1


def test_singular_values_returned():
    L = 3
    mps = _product_mps(L)
    _, S = apply_two_site_gate(mps, _CNOT(), site_i=0)
    assert isinstance(S, np.ndarray)
    assert S.ndim == 1
    assert len(S) >= 1
    assert np.all(S >= 0)


# ---------------------------------------------------------------------------
# Inplace vs copy
# ---------------------------------------------------------------------------

def test_inplace_modifies_original():
    L = 3
    mps = _product_mps(L)
    original_id = id(mps)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=0, inplace=True)
    assert id(result) == original_id


def test_not_inplace_returns_new_object():
    L = 3
    mps = _product_mps(L)
    original_id = id(mps)
    result, _ = apply_two_site_gate(mps, _CNOT(), site_i=0, inplace=False)
    assert id(result) != original_id


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_site_out_of_range_raises():
    L = 3
    mps = _product_mps(L)
    with pytest.raises(ValueError, match="site_i"):
        apply_two_site_gate(mps, _CNOT(), site_i=L - 1)


def test_wrong_gate_shape_raises():
    L = 3
    mps = _product_mps(L)
    bad_gate = np.eye(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="Gate must have shape"):
        apply_two_site_gate(mps, bad_gate, site_i=0)