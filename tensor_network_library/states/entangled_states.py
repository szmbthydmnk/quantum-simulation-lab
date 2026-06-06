from __future__ import annotations

from typing import Tuple
import numpy as np

from tensor_network_library.core.mps import MPS


def _basis_index(bits: np.ndarray) -> int:
    """
    Map a {0,1}^L bitstring to the corresponding computational-basis index.

    bits[0] is the most significant qubit (leftmost).
    """
    bits = np.asarray(bits, dtype=int).reshape(-1)
    idx = 0
    for b in bits:
        if b not in (0, 1):
            raise ValueError(f"Bits must be 0 or 1, got {b!r}")
        idx = (idx << 1) | int(b)
    return idx


def _validate_chain(L: int, physical_dims: int) -> None:
    if L <= 0:
        raise ValueError("L must be a positive integer.")
    if physical_dims != 2:
        # We can support general d later; for TEBD v2 we only need qubits.
        raise ValueError(
            "Entangled state helpers currently support qubits only (physical_dims=2)."
        )


def w_statevector(
    L: int,
    *,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    r"""
    Dense statevector for the L-qubit W state

        |W_L⟩ = (1/√L) Σ_{k=0}^{L-1} |0…010…0⟩_k,

    i.e., equal superposition of all computational basis states with
    a single excitation.
    """
    _validate_chain(L=L, physical_dims=2)

    dim = 2**L
    psi = np.zeros(dim, dtype=dtype)

    if L == 0:
        raise ValueError("Cannot build W-state with L = 0.")

    amp = 1.0 / np.sqrt(float(L))

    bits = np.zeros(L, dtype=int)
    for k in range(L):
        bits.fill(0)
        bits[k] = 1
        idx = _basis_index(bits)
        psi[idx] = amp

    return psi.astype(dtype, copy=False)


def ghz_statevector(L: int,
                    *,
                    dtype: np.dtype = np.complex128,
                    ) -> np.ndarray:
    """
    Dense statevector for the L-qubit GHZ state

        |GHZ_L⟩ = (|0…0⟩ + |1…1⟩) / √2.
    """
    
    _validate_chain(L=L, physical_dims=2)

    dim = 2**L
    psi = np.zeros(dim, dtype=dtype)

    amp = 1.0 / np.sqrt(2.0)

    bits_zero = np.zeros(L, dtype=int)
    bits_one = np.ones(L, dtype=int)

    k0 = _basis_index(bits_zero)
    k1 = _basis_index(bits_one)

    psi[k0] = amp
    psi[k1] = amp

    return psi.astype(dtype, copy=False)


def bell_statevector(L: int = 2,
                     which: str = "phi+",
                     pair: Tuple[int, int] = (0, 1), *,
                     dtype: np.dtype = np.complex128,
                     ) -> np.ndarray:
    """
    Dense statevector for a Bell pair embedded in an L-qubit chain.

    The Bell pair lives on sites `pair = (i, j)`; all other qubits are |0⟩.
    Sites are 0-based, with site 0 the leftmost qubit.

    Supported `which` labels (case-insensitive):

        "phi+" : (|00⟩ + |11⟩) / √2
        "phi-" : (|00⟩ - |11⟩) / √2
        "psi+" : (|01⟩ + |10⟩) / √2
        "psi-" : (|01⟩ - |10⟩) / √2

    When embedded into an L-site chain, these act on qubits i and j,
    with all other qubits frozen in |0⟩.

    Returns
    -------
    psi : np.ndarray, shape (2**L,)
        Normalized statevector.
    """
    
    _validate_chain(L = L, physical_dims = 2)

    i, j = pair
    i = int(i)
    j = int(j)

    if not (0 <= i < L and 0 <= j < L):
        raise ValueError(f"pair={pair!r} must have 0 <= i < j < L, got L={L}")
    if i == j:
        raise ValueError("Bell pair sites must be distinct.")
    if i > j:
        i, j = j, i

    which_key = which.strip().lower()
    if which_key not in {"phi+", "phi-", "psi+", "psi-"}:
        raise ValueError(
            f"Unsupported Bell label {which!r}; "
            "expected one of 'phi+', 'phi-', 'psi+', 'psi-'."
        )

    dim = 2**L
    psi = np.zeros(dim, dtype=dtype)

    bits = np.zeros(L, dtype=int)

    amp = 1.0 / np.sqrt(2.0)

    if which_key.startswith("phi"):
        # |00> ± |11> on (i,j)
        # configuration 1: all zeros
        bits.fill(0)
        k1 = _basis_index(bits)
        # configuration 2: ones on i and j
        bits.fill(0)
        bits[i] = 1
        bits[j] = 1
        k2 = _basis_index(bits)

        if which_key == "phi+":
            psi[k1] = amp
            psi[k2] = amp
        else:  # "phi-"
            psi[k1] = amp
            psi[k2] = -amp

    else:
        # psi±: |01> ± |10> on (i,j)
        bits.fill(0)
        bits[i] = 0
        bits[j] = 1
        k1 = _basis_index(bits)

        bits.fill(0)
        bits[i] = 1
        bits[j] = 0
        k2 = _basis_index(bits)

        if which_key == "psi+":
            psi[k1] = amp
            psi[k2] = amp
        else:  # "psi-"
            psi[k1] = amp
            psi[k2] = -amp

    return psi.astype(dtype, copy=False)

def bell_mps(L: int = 2,
             which: str = "phi+",
             pair: Tuple[int, int] = (0, 1), *,
             name: str = "Bell",
             dtype: np.dtype = np.complex128,
             ) -> MPS:
    """
    Convenience wrapper: build an MPS for a Bell pair embedded in an L-qubit chain.
    
    See 'bell_statevector' for semantics.
    """
    
    psi = bell_statevector(L = L, which = which, pair = pair, dtype=dtype)
    
    return MPS.from_statevector(psi,
                                physical_dims=2,
                                name = name,
                                truncation = None,
                                absorb = "right",
                                normalize = True,
                                dtype = dtype,
                                )
    
    
def ghz_mps(L: int,
            *,
            name: str = "GHZ",
            dtype: np.dtype = np.complex128,
            ) -> MPS:
    """
    Convenience wrapper: build an MPS for the L-qubit GHZ state.
    """
    
    psi = ghz_statevector(L=L, dtype=dtype)
    
    return MPS.from_statevector(psi,
                                physical_dims=2,
                                name=name,
                                truncation=None,
                                absorb="right",
                                normalize=True,
                                dtype=dtype,
                                )


def w_mps(L: int,
          *,
          name: str = "W",
          dtype: np.dtype = np.complex128,) -> MPS:

    """
    Convenience wrapper: build an MPS for the L-qubit W state.
    """
    
    psi = w_statevector(L=L, dtype=dtype)
    
    return MPS.from_statevector(psi,
                                physical_dims=2,
                                name=name,
                                truncation=None,
                                absorb="right",
                                normalize=True,
                                dtype=dtype,
                                )