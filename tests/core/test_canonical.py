#import numpy as np
#from tensor_network_library.core.mps import MPS
#from tensor_network_library.core.mpo import MPO
#from tensor_network_library.core.canonical import left_canonicalize
##from .utils import is_left_orthonormal  # or keep helper in same file
#from tensor_network_library.core.tensor import Tensor
#
#def test_left_canonical_preserves_state():
#    state = [0, 1, 0]
#    mps = MPS.from_product_state(state, physical_dims=2)
#    mpo = MPO.identity(len(state), physical_dims=2)
#    mps = mpo.apply(mps)  # make it slightly less trivial if you want
#
#    psi_before = mps.to_dense()
#    left_canonicalize(mps)
#    psi_after = mps.to_dense()
#
#    assert np.allclose(psi_before, psi_after)
#
#def is_left_orthonormal(A: np.ndarray, atol=1e-10) -> bool:
#    """
#    Check left-orthonormality of a single MPS tensor A with shape (chi_L, d, chi_R).
#    """
#    chi_L, d, chi_R = A.shape
#
#    # reshape to matrix with rows = (alpha_L, s), cols = alpha_R
#    A_mat = A.reshape(chi_L * d, chi_R)
#
#    # compute A^dagger A on the right bond space
#    gram = A_mat.conj().T @ A_mat  # shape (chi_R, chi_R)
#
#    return np.allclose(gram, np.eye(chi_R, dtype=A.dtype), atol=atol)
#
#def test_left_canonical_preserves_state_and_orthonormality():
#    state = [0, 1, 0, 1]
#    mps = MPS.from_product_state(state, physical_dims=2)
#
#    # Make the state a bit less trivial: apply identity or a simple MPO later
#    mpo = MPO.identity(len(state), physical_dims=2)
#    mps = mpo.apply(mps)
#
#    psi_before = mps.to_dense()
#    left_canonicalize(mps)
#    psi_after = mps.to_dense()
#
#    # 1) State is preserved
#    assert np.allclose(psi_before, psi_after)
#
#    # 2) All but the last site are left-orthonormal
#    for i, A in enumerate(mps.tensors[:-1]):
#        assert is_left_orthonormal(A.data)
#        
#def test_left_canonicalization_on_random_mps():
#    L = 4
#    d = 2
#    tensors = []
#    for _ in range(L):
#        data = (np.random.randn(1, d, 1) + 1j * np.random.randn(1, d, 1))
#        t = Tensor(data).normalize()  # use your library normalization
#        tensors.append(t)
#    mps = MPS(tensors)
#
#    psi_before = mps.to_dense()
#    left_canonicalize(mps)
#    psi_after = mps.to_dense()
#
#    assert np.allclose(psi_before, psi_after)
#
#    for A in mps.tensors[:-1]:
#        assert is_left_orthonormal(A.data)