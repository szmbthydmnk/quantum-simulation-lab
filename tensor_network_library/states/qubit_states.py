#

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
    if key in {"i", "I", "+i", "y+", "|i>"}:
        return qubit_pauli_eigenstates("y+")
    if key in {"-i", "-I", "y-", "|-i>"}:
        return qubit_pauli_eigenstates("y-")

    # Equator by angle:
    low = key.lower().replace(" ", "")
    if low.startswith("phi=") or low.startswith("phi:"):
        expr = low.split("=", 1)[1] if "phi=" in low else low.split(":", 1)[1]
        phi = _parse_angle(expr)
        return equator_state(phi)

    if key in {"t", "T"}:
        return qubit_t_type_magic_states(0)


    # Indexed magic families:
    if len(key) >= 2 and (key[0] in {"t", "T"}) and key[1:].isdigit():
        return qubit_t_type_magic_states(int(key[1:]))

    if len(key) >= 2 and (key[0] in {"h", "H"}) and key[1:].isdigit():
        return qubit_h_type_magic_states(int(key[1:]))

    raise ValueError(f"Unknown qubit state label: {label!r}")
    

def qubit_pauli_eigenstates(key: str) -> np.ndarray :
    if key == "x+":
        return _norm(np.array([1.0, 1.0], dtype = np.complex128))
    if key == "x-":
        return _norm(np.array([1.0, -1.0], dtype = np.complex128))
    if key == "y+":
        return _norm(np.array([1.0, 1.0j], dtype = np.complex128))
    if key == "y-":
        return _norm(np.array([1.0, -1.0j], dtype = np.complex128))
    if key == "z+":
        return np.array([1.0, 0.0], dtype = np.complex128)
    if key == "z-":
        return np.array([0.0, 1.0], dtype = np.complex128)
    
    raise ValueError(f"Unknown qubit pauli egienstate: {key}")


def qubit_hadamard_eigenstates(sign: str = '+') -> np.ndarray:
    # Eigenstates of H with eigenvalues plus or minus 1
    # |H+> = cos(pi/8)|0> + sin(pi/8)|1>
    # |H-> = cos(pi/8)|0> - sin(pi/8)|1>
    c = np.cos(np.pi / 8.0)
    s = np.sin(np.pi / 8.0)

    if sign in {"+", "plus", "+1", "p"}:
        return _norm(np.array([c, s], dtype=np.complex128))
    elif sign in {"-", "minus", "-1", "m"}:
        return _norm(np.array([s, -c], dtype=np.complex128))
    raise ValueError(f"Unknown Hadamard sign: {sign!r}")
    

def equator_state(phi: float) -> np.ndarray:
    # (|0> + e^{i phi} |1>) / sqrt(2)
    return _norm(np.array([1.0, np.exp(1j * phi)], dtype=np.complex128))
    

def qubit_t_type_magic_states(k: int) -> np.ndarray:
    """
    8 T-type states: Bloch vectors at cube vertices (±1,±1,±1)/sqrt(3). [web:281]
    """
    k = int(k)
    if not (0 <= k <= 7):
        raise ValueError("T-type index k must be in {0,...,7}.")

    sx = 1.0 if ((k >> 0) & 1) == 0 else -1.0
    sy = 1.0 if ((k >> 1) & 1) == 0 else -1.0
    sz = 1.0 if ((k >> 2) & 1) == 0 else -1.0

    a = np.array([sx, sy, sz], dtype=float) / np.sqrt(3.0)
    return _state_from_bloch(a)


def qubit_h_type_magic_states(k: int) -> np.ndarray:
    """
    H-type family from permutations/signs of (1,1,0)/sqrt(2). [web:281]
    This yields 12 distinct Bloch directions:
      (±1,±1,0), (±1,0,±1), (0,±1,±1) all /sqrt(2).
    """
    vecs: list[np.ndarray] = []
    base_patterns = [
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]

    for bx, by, bz in base_patterns:
        for s1, s2 in signs:
            if bz == 0.0:
                a = np.array([s1 * bx, s2 * by, 0.0], dtype=float)
            elif by == 0.0:
                a = np.array([s1 * bx, 0.0, s2 * bz], dtype=float)
            else:
                a = np.array([0.0, s1 * by, s2 * bz], dtype=float)
            vecs.append(a / np.sqrt(2.0))

    k = int(k)
    if not (0 <= k < len(vecs)):
        raise ValueError(f"H-type index k must be in {{0,...,{len(vecs)-1}}}.")
    return _state_from_bloch(vecs[k])


def qubit_max_magic_states(k: int) -> np.ndarray:
    """
    Alias for the 8 maximal-magic (T-type) qubit states.
    """
    return qubit_t_type_magic_states(k)


def qubit_states(labels: Iterable[str]) -> list[np.ndarray]:
    return [qubit_state(x) for x in labels]


# Put near the bottom of the module
def available_qubit_state_labels() -> list[str]:
    labels = []

    # Pauli / common aliases
    labels += ["0", "1", "+", "-", "i", "-i", "I", "-I"]

    # Hadamard eigenstates (and your "H" alias)
    labels += ["h+", "h-", "H+", "H-", "H"]

    # T-type and H-type indexed families
    labels += [f"t{k}" for k in range(8)]
    labels += [f"h{k}" for k in range(12)]

    # Equator syntax (examples, since it's infinite)
    labels += ["phi=pi/4", "phi:pi/7"]

    return labels


def print_available_qubit_states() -> None:
    labs = available_qubit_state_labels()
    print("Available qubit state labels:")
    for s in labs:
        print(f"  {s}")
