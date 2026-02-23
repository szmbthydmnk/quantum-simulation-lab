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
<<<<<<< Updated upstream
    
=======
    
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

def test_svd_truncation_respects_cutoff_and_max_bond_dim():
    rng = np.random.default_rng(0)
    shape = (4, 6)  # simple 2-index tensor (matrix)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    A = Tensor(data)

    # Policy: keep s^2 >= cutoff, but never more than max_bond_dim
    policy = TruncationPolicy(max_bond_dim=3, cutoff=0.2, strict=False)

    U_full, S_full, V_full = A.svd_decomposition(left_indices=[0], right_indices=[1])
    U_trunc, S_trunc, V_trunc = A.svd(left_indices=[0], right_indices=[1], policy=policy)

    # 1) Truncated singular values are a prefix of the full ones
    assert np.allclose(S_trunc, S_full[: len(S_trunc)])

    # 2) All kept singular values satisfy the cutoff
    assert np.all(S_trunc**2 >= policy.cutoff)

    # 3) Either we've hit max_bond_dim, or the next singular value would violate cutoff
    if len(S_trunc) < min(len(S_full), policy.max_bond_dim):
        assert S_full[len(S_trunc)]**2 < policy.cutoff
    else:
        assert len(S_trunc) <= policy.max_bond_dim

    # 4) Shapes of U and V truncate along the bond dimension only
    chi = len(S_trunc)
    assert U_trunc.data.shape == (shape[0], chi)
    assert V_trunc.data.shape == (chi, shape[1])


def test_svd_truncation_reconstructs_approx_tensor():
    rng = np.random.default_rng(1)
    shape = (3, 4, 5)
    data = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    A = Tensor(data)

    # Use a relaxed cutoff so we drop at least one singular value
    policy = TruncationPolicy(max_bond_dim=10, cutoff=0.05, strict=False)

    U_trunc, S_trunc, V_trunc = A.svd(left_indices=[0], right_indices=[1, 2], policy=policy)

    # Reconstruct approximate tensor from truncated SVD
    left_indices = [0]
    right_indices = [1, 2]
    left_dim = int(np.prod([shape[i] for i in left_indices]))
    right_dim = int(np.prod([shape[i] for i in right_indices]))
    chi = len(S_trunc)

    U_mat = U_trunc.data.reshape(left_dim, chi)
    V_mat = V_trunc.data.reshape(chi, right_dim)
    mat_approx = U_mat @ np.diag(S_trunc) @ V_mat
    data_approx = mat_approx.reshape(shape)

    # Error should be no larger than the discarded singular values
    U_full, S_full, V_full = A.svd_decomposition(left_indices=[0], right_indices=[1, 2])
    discarded = S_full[chi:]
    frob_disc = np.linalg.norm(discarded)

    err = np.linalg.norm(data - data_approx)
    assert err <= frob_disc + 1e-10
>>>>>>> Stashed changes
