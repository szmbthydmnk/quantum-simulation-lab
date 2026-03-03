"""Tests for the Tensor class with Index-based connectivity."""

import pytest
import numpy as np
from tensor_network_library.core.tensor import Tensor
from tensor_network_library.core.index import Index
from tensor_network_library.core.policy import TruncationPolicy


class TestTensorCreation:
    """Test Tensor creation and basic properties."""
    
    def test_tensor_creation_with_auto_indices(self):
        """Test creating a tensor with auto-generated indices."""
        data = np.random.randn(2, 3, 4)
        tensor = Tensor(data)
        
        assert tensor.shape == (2, 3, 4)
        assert tensor.ndim == 3
        assert len(tensor.indices) == 3
        assert all(isinstance(ind, Index) for ind in tensor.indices)
    
    def test_tensor_creation_with_indices(self):
        """Test creating a tensor with provided indices."""
        data = np.random.randn(2, 3, 4)
        inds = [
            Index(dim=2, name="i0"),
            Index(dim=3, name="i1"),
            Index(dim=4, name="i2"),
        ]
        tensor = Tensor(data, indices=inds)
        
        assert tensor.shape == (2, 3, 4)
        assert tensor.indices == inds
    
    def test_tensor_creation_mismatch_dims(self):
        """Test that mismatched index dimensions raise an error."""
        data = np.random.randn(2, 3, 4)
        inds = [
            Index(dim=2, name="i0"),
            Index(dim=5, name="i1"),  # Mismatch!
            Index(dim=4, name="i2"),
        ]
        
        with pytest.raises(AssertionError):
            Tensor(data, indices=inds)


class TestTensorArithmetic:
    """Test basic tensor arithmetic operations."""
    
    def test_tensor_scalar_multiply(self):
        """Test scalar multiplication."""
        data = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        tensor = Tensor(data)
        result = tensor * 2.0
        
        assert np.allclose(result.data, data * 2.0)
    
    def test_tensor_power(self):
        """Test tensor exponentiation."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float64)
        tensor = Tensor(data)
        result = tensor ** 2
        
        assert np.allclose(result.data, data ** 2)
    
    def test_tensor_conjugate(self):
        """Test complex conjugation."""
        data = np.array([[1 + 1j, 2 - 1j]], dtype=np.complex128)
        tensor = Tensor(data)
        result = tensor.conj()
        
        assert np.allclose(result.data, np.conj(data))


class TestTensorReshape:
    """Test reshape operations."""
    
    def test_reshape_2d_to_1d(self):
        """Test reshaping a 2D tensor to 1D."""
        data = np.arange(6).reshape(2, 3)
        tensor = Tensor(data)
        reshaped = tensor.reshape((6,))
        
        assert reshaped.shape == (6,)
        assert np.allclose(reshaped.data, np.arange(6))
    
    def test_reshape_3d_to_2d(self):
        """Test reshaping a 3D tensor to 2D."""
        data = np.arange(24).reshape(2, 3, 4)
        tensor = Tensor(data)
        reshaped = tensor.reshape((6, 4))
        
        assert reshaped.shape == (6, 4)
        assert np.allclose(reshaped.data, data.reshape(6, 4))


class TestTensorTranspose:
    """Test transposition and permutation."""
    
    def test_transpose_default(self):
        """Test transpose with default (reverse) order."""
        data = np.arange(24).reshape(2, 3, 4)
        tensor = Tensor(data)
        transposed = tensor.transpose()
        
        expected = np.transpose(data)
        assert np.allclose(transposed.data, expected)
    
    def test_transpose_custom_axes(self):
        """Test transpose with custom axes."""
        data = np.arange(24).reshape(2, 3, 4)
        tensor = Tensor(data)
        transposed = tensor.transpose((2, 0, 1))
        
        expected = np.transpose(data, (2, 0, 1))
        assert np.allclose(transposed.data, expected)
    
    def test_permute_by_inds(self):
        """Test permuting tensor to match target index order."""
        inds = [
            Index(dim=2, name="i0"),
            Index(dim=3, name="i1"),
            Index(dim=4, name="i2"),
        ]
        data = np.arange(24).reshape(2, 3, 4)
        tensor = Tensor(data, indices=inds)
        
        # Target order: i2, i0, i1
        target_inds = [inds[2], inds[0], inds[1]]
        permuted = tensor.permute_by_inds(target_inds)
        
        expected = np.transpose(data, (2, 0, 1))
        assert np.allclose(permuted.data, expected)
        assert permuted.indices == target_inds


class TestTensorContraction:
    """Test tensor contractions."""
    
    def test_contract_matrices(self):
        """Test contracting two matrices along shared dimension."""
        # A: shape (2, 3), B: shape (3, 4)
        # Contract A[1] with B[0] -> result (2, 4)
        A_data = np.arange(6).reshape(2, 3).astype(np.complex128)
        B_data = np.arange(12).reshape(3, 4).astype(np.complex128)
        
        A = Tensor(A_data)
        B = Tensor(B_data)
        
        result = A.contract(B, axes=([[1], [0]]))
        expected = np.tensordot(A_data, B_data, axes=([1], [0]))
        
        assert np.allclose(result.data, expected)
    
    def test_contract_multidim(self):
        """Test contracting higher-dimensional tensors."""
        # A: (2, 3, 4), B: (4, 5, 6)
        # Contract A[2] with B[0] -> result (2, 3, 5, 6)
        A_data = np.random.randn(2, 3, 4).astype(np.complex128)
        B_data = np.random.randn(4, 5, 6).astype(np.complex128)
        
        A = Tensor(A_data)
        B = Tensor(B_data)
        
        result = A.contract(B, axes=([[2], [0]]))
        expected = np.tensordot(A_data, B_data, axes=([2], [0]))
        
        assert np.allclose(result.data, expected)


class TestTensorQRDecomposition:
    """Test QR decomposition."""
    
    def test_qr_basic(self):
        """Test QR decomposition of a random matrix."""
        data = np.random.randn(4, 6).astype(np.complex128)
        tensor = Tensor(data)
        
        Q, R = tensor.qr_decomposition([0], [1])
        
        # Reconstruct and compare
        reconstructed = Q.data @ R.data
        assert np.allclose(reconstructed, data)
        
        # Check orthogonality of Q
        Q_conj_T = np.conj(Q.data.T)
        assert np.allclose(Q_conj_T @ Q.data, np.eye(Q.data.shape[1]))
    
    def test_qr_3d_tensor(self):
        """Test QR decomposition grouping multiple axes."""
        data = np.random.randn(2, 3, 4, 5).astype(np.complex128)
        tensor = Tensor(data)

        # Group axes 0,1 on left, axes 2,3 on right
        Q, R = tensor.qr_decomposition([0, 1], [2, 3])

        # Shapes: Q = (2, 3, k), R = (k, 4, 5)
        k = Q.shape[2]
        assert Q.shape == (2, 3, k)
        assert R.shape == (k, 4, 5) 

        # Verify indices are properly linked
        assert Q.indices[-1] == R.indices[0]  # Bond index shared

        # Verify QR reconstruction
        Q_mat = Q.data.reshape(6, k)
        R_mat = R.data.reshape(k, 20)
        reconstructed = Q_mat @ R_mat
        original_flat = tensor.data.transpose(0, 1, 2, 3).reshape(6, 20)

        np.testing.assert_allclose(reconstructed, original_flat, atol=1e-10)



class TestTensorSVDDecomposition:
    """Test SVD decomposition."""
    
    def test_svd_basic(self):
        """Test SVD of a random matrix."""
        data = np.random.randn(4, 6).astype(np.complex128)
        tensor = Tensor(data)
        
        U, S, Vh = tensor.svd_decomposition([0], [1])
        
        # Reconstruct
        reconstructed = U.data @ np.diag(S.data) @ Vh.data
        assert np.allclose(reconstructed, data)
        
        # Check singular values are non-negative and sorted
        s_vals = np.asarray(S.data)
        assert np.all(s_vals >= -1e-10)  # Account for numerical precision
        assert np.all(s_vals[:-1] >= s_vals[1:])  # Sorted descending
    
    def test_svd_with_truncation(self):
        """Test SVD with truncation policy."""
        data = np.random.randn(4, 6).astype(np.complex128)
        tensor = Tensor(data)
        
        policy = TruncationPolicy(max_bond_dim=2, cutoff = 0)
        U, S, Vh = tensor.svd([0], [1], policy=policy)
        
        # Bond dimension should be truncated
        assert U.shape[-1] == 2
        assert Vh.shape[0] == 2
        assert len(S.data) == 2


class TestTensorNorm:
    """Test norm operations."""
    
    def test_norm(self):
        """Test computing the Frobenius norm."""
        data = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        tensor = Tensor(data)
        
        norm = tensor.norm()
        expected_norm = np.linalg.norm(data)
        assert np.isclose(norm, expected_norm)
    
    def test_normalize(self):
        """Test normalizing a tensor."""
        data = np.random.randn(2, 3, 4).astype(np.complex128)
        tensor = Tensor(data)
        
        normalized = tensor.normalize()
        assert np.isclose(normalized.norm(), 1.0)
    
    def test_normalize_zero_tensor_raises(self):
        """Test that normalizing a zero tensor raises an error."""
        data = np.zeros((2, 3))
        tensor = Tensor(data)
        
        with pytest.raises(ValueError):
            tensor.normalize()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])