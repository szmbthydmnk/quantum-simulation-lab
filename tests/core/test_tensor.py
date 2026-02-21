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
    
def test_svd_decomposition_reconstructs_tensor():
    # Random complex tensor with 3 indices
    rng = np.random.default_rng(123)
    shape = (3, 4, 5)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    A = Tensor(data)

    # Choose a bipartition: (0) | (1, 2)
    left_indices = [0]
    right_indices = [1, 2]

    U, S, V = A.svd_decomposition(left_indices=left_indices, right_indices=right_indices)

    # Check singular values are 1D
    assert S.ndim == 1

    # Flatten U and V back to matrix form
    left_dim = int(np.prod([shape[i] for i in left_indices]))
    right_dim = int(np.prod([shape[i] for i in right_indices]))
    chi = S.shape[0]

    U_mat = U.data.reshape(left_dim, chi)
    V_mat = V.data.reshape(chi, right_dim)

    # Reconstruct matrix and then tensor
    mat_reconstructed = U_mat @ np.diag(S) @ V_mat
    data_reconstructed = mat_reconstructed.reshape(shape)

    # Check reconstruction
    assert data_reconstructed.shape == shape
    assert np.allclose(data_reconstructed, data, atol=1e-10, rtol=1e-10)
    
def test_svd_decomposition_detects_rank_one():
    rng = np.random.default_rng(321)
    u = rng.normal(size=6) + 1j * rng.normal(size=6)
    v = rng.normal(size=20) + 1j * rng.normal(size=20)
    mat = np.outer(u, v)
    shape = (2, 3, 4, 5)  # 6 × 20
    data = mat.reshape(shape)
    A = Tensor(data)

    U, S, V = A.svd_decomposition(left_indices=[0, 1], right_indices=[2, 3])
    S = np.asarray(S)
    # Only the first singular value should be significant
    assert S[0] > 1e-6
    assert np.all(S[1:] < 1e-10)