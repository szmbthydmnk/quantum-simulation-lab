"""Entangled multi-qubit reference states."""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    """Return normalised copy of a 1-D complex vector."""
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalised.")
    return v / n


# ------------------------------------------------------------------
# Bell states (2 qubits) — legacy simple API
# ------------------------------------------------------------------

def bell_state(label: str = "phi+") -> np.ndarray:
    """
    Return one of the four Bell states as a 4-element complex vector.

    Labels (case-insensitive, spaces/underscores ignored):
      "phi+"  |Phi+> = (|00> + |11>) / sqrt(2)   [default]
      "phi-"  |Phi-> = (|00> - |11>) / sqrt(2)
      "psi+"  |Psi+> = (|01> + |10>) / sqrt(2)
      "psi-"  |Psi-> = (|01> - |10>) / sqrt(2)
    """
    key = label.strip().lower().replace(" ", "").replace("_", "")

    sqrt2 = np.sqrt(2.0)
    states = {
        "phi+": np.array([1, 0, 0, 1], dtype=np.complex128) / sqrt2,
        "phi-": np.array([1, 0, 0, -1], dtype=np.complex128) / sqrt2,
        "psi+": np.array([0, 1, 1, 0], dtype=np.complex128) / sqrt2,
        "psi-": np.array([0, 1, -1, 0], dtype=np.complex128) / sqrt2,
    }
    if key not in states:
        raise ValueError(
            f"Unknown Bell state label {label!r}. "
            f"Choose from: {list(states.keys())}"
        )
    return states[key].copy()


def all_bell_states() -> dict[str, np.ndarray]:
    """Return all four Bell states in a dict."""
    return {lbl: bell_state(lbl) for lbl in ("phi+", "phi-", "psi+", "psi-")}


# ------------------------------------------------------------------
# Bell statevector — embedded into an L-qubit chain
# ------------------------------------------------------------------

def bell_statevector(
    L: int,
    which: str = "phi+",
    pair: Tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Return a Bell state embedded in an L-qubit chain as a dense statevector
    of length 2**L.  All qubits outside ``pair`` are in |0>.

    Parameters
    ----------
    L     : Total number of qubits.
    which : One of "phi+", "phi-", "psi+", "psi-".
    pair  : (i, j) with i < j < L — the two qubits that are entangled.

    Returns
    -------
    np.ndarray of shape (2**L,), dtype complex128, normalised.
    """
    if L < 2:
        raise ValueError("L must be >= 2.")
    i, j = int(pair[0]), int(pair[1])
    if not (0 <= i < j < L):
        raise ValueError(f"pair must satisfy 0 <= i < j < L, got pair={pair}, L={L}.")

    bell_2q = bell_state(which)  # shape (4,)

    dim = 2 ** L
    psi = np.zeros(dim, dtype=np.complex128)

    # Iterate over the 4 two-qubit basis states |s_i, s_j>
    for two_idx in range(4):
        si = (two_idx >> 1) & 1   # bit 1 -> qubit i
        sj = (two_idx >> 0) & 1   # bit 0 -> qubit j
        amp = bell_2q[two_idx]
        if amp == 0:
            continue
        # Build the full L-qubit basis index with si at position i, sj at j, 0 elsewhere
        full_idx = (si << i) | (sj << j)
        psi[full_idx] += amp

    return psi


# ------------------------------------------------------------------
# GHZ states
# ------------------------------------------------------------------

def ghz_state(n: int) -> np.ndarray:
    """
    Return the n-qubit GHZ state (|0...0> + |1...1>) / sqrt(2) as a
    1-D complex vector of length 2**n.
    """
    if n < 2:
        raise ValueError("GHZ state requires n >= 2.")
    dim = 2 ** n
    v = np.zeros(dim, dtype=np.complex128)
    v[0] = 1.0 / np.sqrt(2.0)
    v[dim - 1] = 1.0 / np.sqrt(2.0)
    return v


def ghz_statevector(L: int) -> np.ndarray:
    """
    Alias for :func:`ghz_state` using the ``L`` keyword used by tests.

    Returns
    -------
    np.ndarray of shape (2**L,), dtype complex128, normalised.
    """
    return ghz_state(L)


# ------------------------------------------------------------------
# W states
# ------------------------------------------------------------------

def w_state(n: int) -> np.ndarray:
    """
    Return the n-qubit W state as a 1-D complex vector of length 2**n.
    """
    if n < 2:
        raise ValueError("W state requires n >= 2.")
    dim = 2 ** n
    v = np.zeros(dim, dtype=np.complex128)
    for k in range(n):
        v[1 << k] = 1.0 / np.sqrt(float(n))
    return v


def w_statevector(L: int) -> np.ndarray:
    """
    Alias for :func:`w_state` using the ``L`` keyword used by tests.

    Returns
    -------
    np.ndarray of shape (2**L,), dtype complex128, normalised.
    """
    return w_state(L)


# ------------------------------------------------------------------
# MPS wrappers
# ------------------------------------------------------------------

def bell_mps(
    L: int,
    which: str = "phi+",
    pair: Tuple[int, int] = (0, 1),
):
    """
    Return a Bell state embedded in an L-qubit chain as an MPS.

    Built via :func:`bell_statevector` + ``MPS.from_statevector``.
    The bond dimension is exact (no truncation).
    """
    from tensor_network_library.core.mps import MPS

    psi = bell_statevector(L=L, which=which, pair=pair)
    return MPS.from_statevector(psi, physical_dims=2, name=f"bell_{which}")


def ghz_mps(L: int):
    """
    Return the L-qubit GHZ state as an MPS.

    Built via :func:`ghz_statevector` + ``MPS.from_statevector``.
    """
    from tensor_network_library.core.mps import MPS

    psi = ghz_statevector(L=L)
    return MPS.from_statevector(psi, physical_dims=2, name="ghz")


def w_mps(L: int):
    """
    Return the L-qubit W state as an MPS.

    Built via :func:`w_statevector` + ``MPS.from_statevector``.
    """
    from tensor_network_library.core.mps import MPS

    psi = w_statevector(L=L)
    return MPS.from_statevector(psi, physical_dims=2, name="w")


# ------------------------------------------------------------------
# Cluster states
# ------------------------------------------------------------------

def cluster_state(
    n: int,
    periodic: bool = False,
) -> np.ndarray:
    """
    Return a 1-D cluster state for n qubits as a dense statevector.

    Algorithm
    ---------
    1. Start from |+>^n.
    2. Apply CZ between all pairs (i, i+1). If periodic=True also apply (n-1, 0).
    """
    if n < 2:
        raise ValueError("Cluster state requires n >= 2.")

    dim = 2 ** n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(float(dim))

    def _cz(state: np.ndarray, i: int, j: int, n_qubits: int) -> np.ndarray:
        result = state.copy()
        for idx in range(2 ** n_qubits):
            si = (idx >> i) & 1
            sj = (idx >> j) & 1
            if si == 1 and sj == 1:
                result[idx] *= -1
        return result

    for i in range(n - 1):
        psi = _cz(psi, i, i + 1, n)
    if periodic:
        psi = _cz(psi, n - 1, 0, n)

    return psi


# ------------------------------------------------------------------
# Dicke states
# ------------------------------------------------------------------

def dicke_state(n: int, k: int) -> np.ndarray:
    """
    Return the Dicke state |D^n_k>, the equal superposition of all
    n-qubit basis states with exactly k ones.
    """
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n, got n={n}, k={k}.")
    dim = 2 ** n
    v = np.zeros(dim, dtype=np.complex128)
    for idx in range(dim):
        if bin(idx).count("1") == k:
            v[idx] = 1.0
    return _norm(v)
