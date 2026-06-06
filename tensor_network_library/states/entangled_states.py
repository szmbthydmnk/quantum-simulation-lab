"""Entangled multi-qubit reference states."""

from __future__ import annotations

import numpy as np
from typing import Union

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
# Bell states (2 qubits)
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
    return {
        lbl: bell_state(lbl)
        for lbl in ("phi+", "phi-", "psi+", "psi-")
    }


# ------------------------------------------------------------------
# GHZ states (n qubits)
# ------------------------------------------------------------------

def ghz_state(n: int) -> np.ndarray:
    """
    Return the n-qubit GHZ state  (|0...0> + |1...1>) / sqrt(2)  as a
    1-D complex vector of length 2**n.
    """
    if n < 2:
        raise ValueError("GHZ state requires n >= 2.")
    dim = 2 ** n
    v   = np.zeros(dim, dtype=np.complex128)
    v[0]      = 1.0 / np.sqrt(2.0)   # |00...0>
    v[dim - 1] = 1.0 / np.sqrt(2.0)  # |11...1>
    return v


# ------------------------------------------------------------------
# W states (n qubits)
# ------------------------------------------------------------------

def w_state(n: int) -> np.ndarray:
    """
    Return the n-qubit W state  (|100...0> + |010...0> + ... + |000...1>) / sqrt(n)
    as a 1-D complex vector of length 2**n.
    """
    if n < 2:
        raise ValueError("W state requires n >= 2.")
    dim = 2 ** n
    v   = np.zeros(dim, dtype=np.complex128)
    for k in range(n):
        # |000...1(at position k)...000>  in little-endian bit ordering
        v[1 << k] = 1.0 / np.sqrt(float(n))
    return v


# ------------------------------------------------------------------
# Cluster states (1-D ring or open chain)
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

    CZ acts on the computational basis as  CZ|s1, s2> = (-1)^{s1 s2} |s1, s2>.

    Returns
    -------
    np.ndarray of shape (2**n,), dtype complex128.
    """
    if n < 2:
        raise ValueError("Cluster state requires n >= 2.")

    # Start from |+>^n: every amplitude = 1/sqrt(2^n)
    dim  = 2 ** n
    psi  = np.ones(dim, dtype=np.complex128) / np.sqrt(float(dim))

    def _cz(state: np.ndarray, i: int, j: int, n_qubits: int) -> np.ndarray:
        """Apply CZ_{i,j} in-place."""
        result = state.copy()
        for idx in range(2 ** n_qubits):
            si = (idx >> i) & 1
            sj = (idx >> j) & 1
            if si == 1 and sj == 1:
                result[idx] *= -1
        return result

    # Apply CZ along the chain
    for i in range(n - 1):
        psi = _cz(psi, i, i + 1, n)
    if periodic:
        psi = _cz(psi, n - 1, 0, n)

    return psi


# ------------------------------------------------------------------
# Dicke states (n qubits, k excitations)
# ------------------------------------------------------------------

def dicke_state(n: int, k: int) -> np.ndarray:
    """
    Return the Dicke state |D^n_k>, the equal superposition of all
    n-qubit basis states with exactly k ones.

    Parameters
    ----------
    n : Number of qubits.
    k : Number of excitations (0 <= k <= n).

    Returns
    -------
    np.ndarray of shape (2**n,), dtype complex128, normalized.
    """
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n, got n={n}, k={k}.")
    dim = 2 ** n
    v   = np.zeros(dim, dtype=np.complex128)
    for idx in range(dim):
        if bin(idx).count("1") == k:
            v[idx] = 1.0
    return _norm(v)
