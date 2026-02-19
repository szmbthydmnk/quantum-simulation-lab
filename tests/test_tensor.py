import numpy as np
from tensor_network_library.core.tensor import Tensor

def test_tensor_copy_and_conj():
    data = np.array([[1+1j, 2-3j]], dtype=np.complex128)
    t = Tensor(data)

    t_copy = t.copy()
    assert np.allclose(t.data, t_copy.data)
    assert t is not t_copy

    t_conj = t.conj()
    assert np.allclose(t_conj.data, np.conjugate(data))

def test_tensor_contract_simple():
    a = Tensor(np.arange(6, dtype=np.complex128).reshape(2, 3))
    b = Tensor(np.ones((3, 4), dtype=np.complex128))

    c = a.contract(b, axes=([1], [0]))
    assert c.shape == (2, 4)

    expected = np.tensordot(a.data, b.data, axes=([1], [0]))
    assert np.allclose(c.data, expected)
    