# tests/states/test_entangled_states.py

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.states.entangled_states import (
    bell_statevector,
    ghz_statevector,
    w_statevector,
    bell_mps,
    ghz_mps,
    w_mps,
)
from tensor_network_library.core.mps import MPS


def _norm(psi: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(psi).reshape(-1)))


# ------------------------
# Dense statevector tests
# ------------------------


@pytest.mark.parametrize("which", ["phi+", "phi-", "psi+", "psi-"])
def test_bell_statevector_L2_matches_definitions(which: str) -> None:
    L = 2
    psi = bell_statevector(L=L, which=which, pair=(0, 1))
    assert psi.shape == (2**L,)
    assert np.isclose(_norm(psi), 1.0)

    # Explicit reference definitions on 2 qubits
    z00 = np.array([1, 0, 0, 0], dtype=np.complex128)
    z01 = np.array([0, 1, 0, 0], dtype=np.complex128)
    z10 = np.array([0, 0, 1, 0], dtype=np.complex128)
    z11 = np.array([0, 0, 0, 1], dtype=np.complex128)

    if which == "phi+":
        ref = (z00 + z11) / np.sqrt(2.0)
    elif which == "phi-":
        ref = (z00 - z11) / np.sqrt(2.0)
    elif which == "psi+":
        ref = (z01 + z10) / np.sqrt(2.0)
    else:  # psi-
        ref = (z01 - z10) / np.sqrt(2.0)

    assert np.allclose(psi, ref)


def test_bell_statevector_embedding_into_longer_chain() -> None:
    L = 4
    # Bell pair on the middle sites (1, 2), others in |0>
    psi = bell_statevector(L=L, which="phi+", pair=(1, 2))

    assert psi.shape == (2**L,)
    assert np.isclose(_norm(psi), 1.0)

    # Only two basis states should have nonzero amplitudes:
    # |0000> and |0110>, each with amplitude 1/sqrt(2)
    amps_nonzero = np.where(np.abs(psi) > 1e-12)[0]
    assert set(amps_nonzero.tolist()) == {0, 0b0110}

    assert np.allclose(
        sorted(np.abs(psi[amps_nonzero])),
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
    )


@pytest.mark.parametrize("L", [2, 3, 5])
def test_ghz_statevector(L: int) -> None:
    psi = ghz_statevector(L=L)
    assert psi.shape == (2**L,)
    assert np.isclose(_norm(psi), 1.0)

    # Only |0...0> and |1...1> should appear
    nz = np.where(np.abs(psi) > 1e-12)[0]
    assert len(nz) == 2

    k0 = 0
    k1 = (1 << L) - 1
    assert set(nz.tolist()) == {k0, k1}

    # Amplitudes equal in magnitude
    mag = np.abs(psi[nz])
    assert np.allclose(mag[0], mag[1])


@pytest.mark.parametrize("L", [2, 3, 5])
def test_w_statevector(L: int) -> None:
    psi = w_statevector(L=L)
    assert psi.shape == (2**L,)
    assert np.isclose(_norm(psi), 1.0)

    nz = np.where(np.abs(psi) > 1e-12)[0]
    # There should be exactly L basis states with a single excitation
    assert len(nz) == L

    # Check that each nonzero configuration has Hamming weight 1
    for idx in nz:
        bits = [(idx >> k) & 1 for k in range(L)]
        assert sum(bits) == 1

    # All nonzero amplitudes have equal magnitude
    mags = np.abs(psi[nz])
    assert np.allclose(mags, mags[0])


# ------------------------
# MPS wrapper tests
# ------------------------


def test_bell_mps_to_dense_matches_vector() -> None:
    L = 4
    psi = bell_statevector(L=L, which="psi-", pair=(1, 2))
    mps = bell_mps(L=L, which="psi-", pair=(1, 2))

    assert isinstance(mps, MPS)
    assert len(mps) == L

    dense = mps.to_dense()
    # Allow for a possible global phase; compare up to a phase factor
    phase = dense[0] / psi[0] if np.abs(psi[0]) > 1e-12 else 1.0
    assert np.allclose(dense, phase * psi)


def test_ghz_mps_to_dense_matches_vector() -> None:
    L = 5
    psi = ghz_statevector(L=L)
    mps = ghz_mps(L=L)

    assert isinstance(mps, MPS)
    assert len(mps) == L

    dense = mps.to_dense()
    phase = dense[0] / psi[0] if np.abs(psi[0]) > 1e-12 else 1.0
    assert np.allclose(dense, phase * psi)


def test_w_mps_to_dense_matches_vector() -> None:
    L = 4
    psi = w_statevector(L=L)
    mps = w_mps(L=L)

    assert isinstance(mps, MPS)
    assert len(mps) == L

    dense = mps.to_dense()
    # W has more nonzero components; estimate phase from first nonzero entry
    nz = np.where(np.abs(psi) > 1e-12)[0]
    ref_idx = int(nz[0])
    phase = dense[ref_idx] / psi[ref_idx]
    assert np.allclose(dense, phase * psi)