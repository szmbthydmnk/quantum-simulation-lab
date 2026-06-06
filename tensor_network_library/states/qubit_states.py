"""
Single-qubit and multi-qubit product-state factories.

Provides convenience constructors for standard computational-basis and
Bloch-sphere states as :class:`~tensor_network_library.core.mps.MPS` objects,
as well as uniform and domain-wall product-state builders used as initialisers
for imaginary-time evolution and DMRG.

State catalogue
---------------
Single-qubit (returned as length-1 MPS or plain ndarray):
    ``zero``, ``one``          – |0⟩, |1⟩  (computational basis)
    ``plus``, ``minus``        – |±⟩ = (|0⟩ ± |1⟩)/√2
    ``plus_i``, ``minus_i``    – |±i⟩ = (|0⟩ ± i|1⟩)/√2

Multi-qubit product states:
    ``all_zeros``, ``all_ones``       – uniform ↑↑…↑, ↓↓…↓
    ``all_plus``                      – uniform |+⟩⊗L (maximal superposition)
    ``neel``                          – |↑↓↑↓…⟩ (classical Néel order)
    ``domain_wall``                   – |↑…↑↓…↓⟩ with configurable wall position
    ``random_product``                – product state with random Bloch angles

Axis conventions:
    site tensor shape : (χ_left=1, d=2, χ_right=1)  for all product states
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import warnings
import re


def _norm(v: np.ndarray) -> np.ndarray:
    """
    Helper.

    Normalise vectors.
    """
    v = np.asarray(v, dtype=np.complex128).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector is not a valid state.")
    
    if n <= 1e-10:
        warnings.warn("State vector norm is very small.", stacklevel=2)
    
    return v / n


def _state_from_bloch(a: np.ndarray) -> np.ndarray:
    """
    Helper.

    Convert a (unit) Bloch Vector a = (a_x, a_y, a_z) to a pure qubit statevector.

    Uses |psi> = cos(theta/2)|0> + exp(i phi) sin(theta/2)|1>,
    where az=cos(theta), ax=sin(theta)cos(phi), ay=sin(theta)sin(phi).
    """

    a = np.asarray(a, dtype=float).reshape(3)

    na = np.linalg.norm(a)
    if na == 0:
        raise ValueError("Bloch vector must be nonzero.")
    
    a = a / na

    ax, ay, az = a
    az = float(np.clip(az, -1.0, 1.0))

    theta = float(np.arccos(az))
    phi = float(np.arctan2(ay, ax))

    v = np.array([np.cos(theta / 2.0), np.exp(1j * phi) * np.sin(theta / 2.0)], dtype=np.complex128)

    return _norm(v)


def _parse_angle(expr: str) -> float:
    """
    Helper. 

    Parse angles like:
      "0.3"
      "pi", "-pi", "pi/4", "-3*pi/8", "7pi/8"
    """
    s = expr.strip().lower().replace(" ", "")
    if s == "":
        raise ValueError("Empty angle expression.")

    # Plain float
    if "pi" not in s:
        return float(s)

    # Normalize "7pi/8" -> "7*pi/8"
    s = re.sub(r"(\d)pi", r"\1*pi", s)

    m = re.fullmatch(
        r"(?P<sgn>[+-])?"
        r"(?:(?P<num>(?:\d+(?:\.\d*)?|\.\d+))\*)?"
        r"pi"
        r"(?:/(?P<den>(?:\d+(?:\.\d*)?|\.\d+)))?",
        s,
    )
    if m is None:
        raise ValueError(f"Unsupported angle expression: {expr!r}")

    sgn = -1.0 if m.group("sgn") == "-" else 1.0
    num = float(m.group("num")) if m.group("num") is not None else 1.0
    den = float(m.group("den")) if m.group("den") is not None else 1.0
    if den == 0.0:
        raise ValueError("Angle expression has zero denominator.")

    return sgn * num * np.pi / den


def qubit_state(label: str) -> np.ndarray:
    """
    Return a normalized 1-qubit statevector for common labels.

    Supported (aliases included):
        X eigenstates: '+', '-', 'x+', 'x-'
        Y eigenstates: 'i', '-i', 'y+', 'y-'
        Z eigenstates: '0', '1', 'z+', 'z-'
        H eigenstates: 'H+', 'H-', 'h-', 'h+'
        H-type magic states: 'H#', 'h#'        (# in 0..11 by this construction)
        T-type magic states: 'T#', 't#'        (# in 0..7)
        Equator state: 'phi=EXPR' or 'phi:EXPR' where EXPR may include pi
    """
    key = label.strip()

    # Hadamard eigenstates:
    if key in {"h+", "H+", "hadamard+", "Hadamard+", "H"}:
        return qubit_hadamard_eigenstates("+")
    if key in {"h-", "H-", "hadamard-", "Hadamard-"}:
        return qubit_hadamard_eigenstates("-")

    # Pauli states:
    if key in {"0", "z+", "|0>"}:
        return qubit_pauli_eigenstates("z+")
    if key in {"1", "z-", "|1>"}:
        return qubit_pauli_eigenstates("z-")
    if key in {"+", "x+", "|+>"}:
        return qubit_pauli_eigenstates("x+")
    if key in {"-", "x-", "|->"}:
        return qubit_pauli_eigenstates("x-")
    if key in {"i", "y+", "|i>"}:
        return qubit_pauli_eigenstates("y+")
    if key in {"-i", "y-", "|-i>"}:
        return qubit_pauli_eigenstates("y-")

    # Magic states (T-type):
    m = re.fullmatch(r"[Tt](\d+)", key)
    if m:
        idx = int(m.group(1))
        return qubit_magic_state(f"T{idx}")

    # Magic states (H-type):
    m = re.fullmatch(r"[Hh](\d+)", key)
    if m:
        idx = int(m.group(1))
        return qubit_magic_state(f"H{idx}")

    # Equator states phi=angle:
    for sep in ("=", ":"):
        if key.startswith("phi" + sep) or key.startswith("PHI" + sep):
            angle_str = key.split(sep, 1)[1]
            phi = _parse_angle(angle_str)
            return qubit_equatorial_state(phi)

    raise ValueError(
        f"Unknown qubit state label: {label!r}. "
        "Supported: '0','1','+','-','i','-i','z+','z-','x+','x-','y+','y-',"
        "'H+','H-','H0'..'H11','T0'..'T7','phi=<angle>'."
    )


def qubit_states(labels: Iterable[str]) -> list[np.ndarray]:
    """
    Return a list of normalized 1-qubit statevectors for a sequence of labels.

    This is the batch counterpart of :func:`qubit_state`.  Each label is
    resolved independently; the result order matches the input order.

    Args:
        labels: Iterable of label strings accepted by :func:`qubit_state`.

    Returns:
        List of 1-D complex128 arrays, one per label.

    Example::

        vecs = qubit_states(["0", "+", "T0", "phi=pi/4"])
    """
    return [qubit_state(lbl) for lbl in labels]


def qubit_pauli_eigenstates(label: str) -> np.ndarray:
    """
    Return normalized eigenstates of Pauli operators.

    Labels:
        'z+' -> |0> = [1, 0]
        'z-' -> |1> = [0, 1]
        'x+' -> |+> = [1, 1]/sqrt(2)
        'x-' -> |-> = [1, -1]/sqrt(2)
        'y+' -> |i> = [1, i]/sqrt(2)
        'y-' -> |-i> = [1, -i]/sqrt(2)
    """
    states = {
        "z+": np.array([1.0, 0.0], dtype=np.complex128),
        "z-": np.array([0.0, 1.0], dtype=np.complex128),
        "x+": np.array([1.0,  1.0], dtype=np.complex128) / np.sqrt(2),
        "x-": np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2),
        "y+": np.array([1.0,  1j],  dtype=np.complex128) / np.sqrt(2),
        "y-": np.array([1.0, -1j],  dtype=np.complex128) / np.sqrt(2),
    }
    if label not in states:
        raise ValueError(f"Unknown Pauli eigenstate label: {label!r}. Expected one of {list(states.keys())}.")
    return states[label]


def qubit_hadamard_eigenstates(sign: str = "+") -> np.ndarray:
    """
    Return the +1 or -1 eigenstate of the Hadamard operator H.

    H = (X + Z) / sqrt(2)

    Eigenvalues: +1 and -1.
    Eigenstates:
        +1: cos(pi/8)|0> + sin(pi/8)|1>
        -1: sin(pi/8)|0> - cos(pi/8)|1>
    """
    theta = np.pi / 8.0
    if sign == "+":
        return np.array([np.cos(theta), np.sin(theta)], dtype=np.complex128)
    elif sign == "-":
        return np.array([np.sin(theta), -np.cos(theta)], dtype=np.complex128)
    else:
        raise ValueError(f"sign must be '+' or '-', got {sign!r}")


def qubit_magic_state(label: str) -> np.ndarray:
    """
    Return a magic state by label.

    Supported labels:
        T-type: 'T0'..'T7'   -> 8 T-type magic states on the great circle
                                   between |+> and |T> = cos(pi/8)|0> + e^{i*pi/4} sin(pi/8)|1>
        H-type: 'H0'..'H11'  -> 12 H-type magic states (vertices of the
                                    cuboctahedron inscribed in the Bloch sphere)

    Note: T0 is the standard T-magic state.
    """
    m_t = re.fullmatch(r"T(\d+)", label)
    m_h = re.fullmatch(r"H(\d+)", label)

    if m_t:
        k = int(m_t.group(1)) % 8
        theta = np.pi / 4.0
        phi   = k * np.pi / 4.0
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return _norm(np.array([c, np.exp(1j * phi) * s], dtype=np.complex128))

    elif m_h:
        k = int(m_h.group(1)) % 12
        bloch_vectors = [
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
            (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
        ]
        return _state_from_bloch(np.array(bloch_vectors[k], dtype=float))

    else:
        raise ValueError(f"Unknown magic state label: {label!r}. Expected 'T0'..'T7' or 'H0'..'H11'.")


def qubit_equatorial_state(phi: float) -> np.ndarray:
    """
    Return the equatorial qubit state at azimuthal angle phi:

        |phi> = (|0> + e^{i*phi}|1>) / sqrt(2)
    """
    return _norm(np.array([1.0, np.exp(1j * phi)], dtype=np.complex128))


def all_zeros_mps(
    L: int,
    d: int = 2,
    dtype: np.dtype = np.complex128,
) -> "MPS":
    """
    Build a product-state MPS |0 0 ... 0> (all sites in the |0> state).

    Args:
        L:     Chain length.
        d:     Physical dimension (default 2).
        dtype: Data type.

    Returns:
        MPS representing |0⟩^{⊗L}.
    """
    from tensor_network_library.core.mps import MPS
    e0 = np.zeros(d, dtype=dtype)
    e0[0] = 1.0
    return MPS.product_state([e0] * L, dtype=dtype, name="all_zeros")


def all_ones_mps(
    L: int,
    d: int = 2,
    dtype: np.dtype = np.complex128,
) -> "MPS":
    """
    Build a product-state MPS |1 1 ... 1> (all sites in the |1> state).

    Args:
        L:     Chain length.
        d:     Physical dimension (default 2).
        dtype: Data type.

    Returns:
        MPS representing |1⟩^{⊗L}.
    """
    from tensor_network_library.core.mps import MPS
    e1 = np.zeros(d, dtype=dtype)
    e1[1 % d] = 1.0
    return MPS.product_state([e1] * L, dtype=dtype, name="all_ones")


def neel_mps(
    L: int,
    dtype: np.dtype = np.complex128,
    start: int = 0,
) -> "MPS":
    """
    Build the classical Néel state |01010101...⟩ or |10101010...⟩.

    Args:
        L:     Chain length.
        dtype: Data type.
        start: 0 starts with |0⟩ (even sites |0⟩, odd sites |1⟩);
               1 starts with |1⟩ (even sites |1⟩, odd sites |0⟩).

    Returns:
        MPS representing the Néel state.
    """
    from tensor_network_library.core.mps import MPS
    d = 2
    states = []
    for i in range(L):
        v = np.zeros(d, dtype=dtype)
        v[(i + start) % 2] = 1.0
        states.append(v)
    return MPS.product_state(states, dtype=dtype, name=f"neel_start{start}")


def domain_wall_mps(
    L: int,
    wall: int | None = None,
    dtype: np.dtype = np.complex128,
) -> "MPS":
    """
    Build a domain-wall state |↑…↑↓…↓⟩.

    The first `wall` sites are in state |0⟩ (↑) and the remaining
    L - wall sites are in state |1⟩ (↓).

    Args:
        L:    Chain length.
        wall: Number of up-spin sites (default L//2).
        dtype: Data type.

    Returns:
        MPS representing the domain-wall state.
    """
    from tensor_network_library.core.mps import MPS
    if wall is None:
        wall = L // 2
    if not (0 <= wall <= L):
        raise ValueError(f"wall={wall} must be in [0, L={L}].")
    d = 2
    states = []
    for i in range(L):
        v = np.zeros(d, dtype=dtype)
        v[0 if i < wall else 1] = 1.0
        states.append(v)
    return MPS.product_state(states, dtype=dtype, name=f"domain_wall_w{wall}")


def random_product_mps(
    L: int,
    seed: int | None = None,
    dtype: np.dtype = np.complex128,
) -> "MPS":
    """
    Build a random product-state MPS.

    Each site is an independent random pure qubit state drawn uniformly
    from the Bloch sphere (Haar measure on SU(2)).

    Args:
        L:    Chain length.
        seed: Optional random seed.
        dtype: Data type.

    Returns:
        MPS representing a random product state.
    """
    from tensor_network_library.core.mps import MPS
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(L):
        v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        v = v.astype(np.complex128)
        v /= np.linalg.norm(v)
        states.append(v.astype(dtype))
    return MPS.product_state(states, dtype=dtype, name="random_product")
