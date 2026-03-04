import numpy as np
import pytest

import tensor_network_library.states.qubit_states as qs  # type: ignore

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
    idx = np.argmax(np.abs(w))
    if np.abs(w[idx]) < atol:
        return np.allclose(v, w, atol=atol, rtol=0)
    phase = v[idx] / w[idx]
    return np.allclose(v, phase * w, atol=atol, rtol=0)


def _bloch_from_state(psi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi, dtype=np.complex128).reshape(2)
    psi = psi / np.linalg.norm(psi)

    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    ax = np.vdot(psi, X @ psi).real
    ay = np.vdot(psi, Y @ psi).real
    az = np.vdot(psi, Z @ psi).real
    return np.array([ax, ay, az], dtype=float)


def test__norm_normalizes():
    v = np.array([3.0 + 0j, 4.0 + 0j])
    u = qs._norm(v)
    assert np.isclose(np.linalg.norm(u), 1.0)


def test__norm_raises_on_zero():
    with pytest.raises(ValueError):
        qs._norm(np.array([0.0, 0.0], dtype=np.complex128))


def test__norm_warns_on_small_norm():
    with pytest.warns(UserWarning):
        qs._norm(np.array([1e-12, 0.0], dtype=np.complex128))


def test__state_from_bloch_raises_on_zero():
    with pytest.raises(ValueError):
        qs._state_from_bloch(np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "a",
    [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
    ],
)
def test__state_from_bloch_roundtrip_bloch(a):
    psi = qs._state_from_bloch(a)
    b = _bloch_from_state(psi)
    a = a / np.linalg.norm(a)
    assert np.allclose(b, a, atol=1e-10, rtol=0)


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("0.3", 0.3),
        ("pi", np.pi),
        ("pi/4", np.pi / 4.0),
        ("3*pi/4", 3.0 * np.pi / 4.0),
        ("7pi/8", 7.0 * np.pi / 8.0),
        ("-pi/2", -np.pi / 2.0),
    ],
)
def test__parse_angle(expr, expected):
    # If you kept the eval-based parser, consider hardening it; these tests
    # reflect the intended supported grammar from your docstring.
    got = qs._parse_angle(expr)
    assert np.isclose(got, expected)


@pytest.mark.parametrize("expr", ["pi**2", "pi//2", "os.system('echo pwn')", ""])
def test__parse_angle_rejects_unsupported(expr):
    with pytest.raises(ValueError):
        qs._parse_angle(expr)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("z+", np.array([1.0, 0.0], dtype=np.complex128)),
        ("z-", np.array([0.0, 1.0], dtype=np.complex128)),
        ("x+", np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)),
        ("x-", np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)),
        ("y+", np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0)),
        ("y-", np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0)),
    ],
)
def test_qubit_pauli_eigenstates(key, expected):
    got = qs.qubit_pauli_eigenstates(key)
    assert _global_phase_equal(got, expected)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_qubit_pauli_eigenstates_unknown_raises():
    with pytest.raises(ValueError):
        qs.qubit_pauli_eigenstates("nope")


def test_qubit_hadamard_eigenstates_are_eigenvectors():
    H = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

    v_plus = qs.qubit_hadamard_eigenstates("+")
    v_minus = qs.qubit_hadamard_eigenstates("-")

    assert np.allclose(H @ v_plus, +v_plus, atol=1e-12, rtol=0)
    assert np.allclose(H @ v_minus, -v_minus, atol=1e-12, rtol=0)
    assert np.isclose(np.vdot(v_plus, v_minus), 0.0, atol=1e-12)


def test_equator_state_matches_definition():
    phi = np.pi / 7.0
    got = qs.equator_state(phi)
    expected = np.array([1.0, np.exp(1j * phi)], dtype=np.complex128) / np.sqrt(2.0)
    assert _global_phase_equal(got, expected)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_qubit_state_pauli_aliases():
    assert _global_phase_equal(qs.qubit_state("0"), qs.qubit_pauli_eigenstates("z+"))
    assert _global_phase_equal(qs.qubit_state("|0>"), qs.qubit_pauli_eigenstates("z+"))
    assert _global_phase_equal(qs.qubit_state("1"), qs.qubit_pauli_eigenstates("z-"))
    assert _global_phase_equal(qs.qubit_state("+"), qs.qubit_pauli_eigenstates("x+"))
    assert _global_phase_equal(qs.qubit_state("-"), qs.qubit_pauli_eigenstates("x-"))
    assert _global_phase_equal(qs.qubit_state("i"), qs.qubit_pauli_eigenstates("y+"))
    assert _global_phase_equal(qs.qubit_state("I"), qs.qubit_pauli_eigenstates("y+"))
    assert _global_phase_equal(qs.qubit_state("-i"), qs.qubit_pauli_eigenstates("y-"))
    assert _global_phase_equal(qs.qubit_state("-I"), qs.qubit_pauli_eigenstates("y-"))


def test_qubit_state_hadamard_aliases():
    assert _global_phase_equal(qs.qubit_state("h+"), qs.qubit_hadamard_eigenstates("+"))
    assert _global_phase_equal(qs.qubit_state("H+"), qs.qubit_hadamard_eigenstates("+"))
    assert _global_phase_equal(qs.qubit_state("h-"), qs.qubit_hadamard_eigenstates("-"))
    assert _global_phase_equal(qs.qubit_state("H-"), qs.qubit_hadamard_eigenstates("-"))
    assert _global_phase_equal(qs.qubit_state("H"), qs.qubit_hadamard_eigenstates("+"))


def test_qubit_state_phi_angle():
    v = qs.qubit_state("phi=pi/4")
    expected = qs.equator_state(np.pi / 4.0)
    assert _global_phase_equal(v, expected)


def test_qubit_t_type_magic_states_count_and_bloch():
    states = [qs.qubit_t_type_magic_states(k) for k in range(8)]
    # Ensure all are normalized and distinct (up to a sign/global phase).
    for v in states:
        assert np.isclose(np.linalg.norm(v), 1.0)

    # Check Bloch magnitudes: components should be ±1/sqrt(3).
    target = 1.0 / np.sqrt(3.0)
    for k, v in enumerate(states):
        a = _bloch_from_state(v)
        assert np.allclose(np.abs(a), target, atol=1e-10, rtol=0)

        # Optional: verify the sign convention matches your bit mapping.
        sx = +1.0 if ((k >> 0) & 1) == 0 else -1.0
        sy = +1.0 if ((k >> 1) & 1) == 0 else -1.0
        sz = +1.0 if ((k >> 2) & 1) == 0 else -1.0
        assert np.allclose(a, np.array([sx, sy, sz]) * target, atol=1e-10, rtol=0)


def test_qubit_t_type_magic_states_invalid_k_raises():
    with pytest.raises(ValueError):
        qs.qubit_t_type_magic_states(-1)
    with pytest.raises(ValueError):
        qs.qubit_t_type_magic_states(8)


def test_qubit_h_type_magic_states_count_and_bloch():
    states = [qs.qubit_h_type_magic_states(k) for k in range(12)]
    for v in states:
        assert np.isclose(np.linalg.norm(v), 1.0)

    target = 1.0 / np.sqrt(2.0)
    for v in states:
        a = _bloch_from_state(v)
        # Exactly one component is (approximately) 0; the other two are ±1/sqrt(2).
        zeros = np.isclose(a, 0.0, atol=1e-10, rtol=0)
        assert zeros.sum() == 1
        nz = a[~zeros]
        assert np.allclose(np.abs(nz), target, atol=1e-10, rtol=0)


def test_qubit_h_type_magic_states_invalid_k_raises():
    with pytest.raises(ValueError):
        qs.qubit_h_type_magic_states(-1)
    with pytest.raises(ValueError):
        qs.qubit_h_type_magic_states(12)


def test_qubit_max_magic_states_is_alias_of_t_type():
    for k in range(8):
        assert _global_phase_equal(qs.qubit_max_magic_states(k), qs.qubit_t_type_magic_states(k))


def test_qubit_states_batch():
    labels = ["0", "+", "i", "t0", "h0", "phi=pi/4"]
    out = qs.qubit_states(labels)
    assert isinstance(out, list)
    assert len(out) == len(labels)
    for v in out:
        assert np.asarray(v).shape == (2,)
        assert np.isclose(np.linalg.norm(v), 1.0)
