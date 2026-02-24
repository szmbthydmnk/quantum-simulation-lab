from __future__ import annotations  # Postpone evaluation of type hints (allows forward references, avoids circular imports)


from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd, qr, eigh
from .policy import TruncationPolicy

ComplexArray = NDArray[np.complex128]


class Tensor:
    """
    A multi dimensional array wrapper with tensor network operations
    
    Attributes:
        data (np.ndarray): The underlying numpy array.
        shape (tuple): Shape/dimensions of the tensor.
        ndim (int): Number of dimensions.
    """

    def __init__(self, 
                 data: ComplexArray,
                 physical_indices: Optional[List[str]] = None,
                 bond_indices: Optional[List[str]] = None):
        """
        Initialize a tensor from a numpy array
        
        Args:
            data: NumPy array
            physical_indices: Names of physical indices (e.g., ['p_1', 'p_2'])
            bond_indices: Names of bond indices (e.g., ['L', 'R'])
        """
        self.data = np.asarray(data, dtype = np.complex128)
        self.physical_indices = physical_indices or []
        self.bond_indices = bond_indices or []

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns shape of the tensor"""
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        """Return number of dimensions"""
        return self.data.ndim

    def copy(self) -> 'Tensor':
        """Create a deep copy of the tensor"""
        return Tensor(self.data.copy(), physical_indices=self.physical_indices.copy(), bond_indices=self.bond_indices.copy())
    
    def conj(self) -> 'Tensor':
        """Return the complex conugate of the tensor"""
        return Tensor(np.conj(self.data), physical_indices=self.physical_indices.copy(), bond_indices=self.bond_indices.copy())
    
    def contract(self, other: 'Tensor', axes: Tuple[List[int], List[int]]) -> 'Tensor':
        """
        Contract two tensors along specified axis
        
        Args:
            other: Tensor to contract with.
            axes: Tuple of two lists specifying axes to contract.
                    axes[0] are axes from self and axes[1] are axes from other.
        
        Returns:
            Contracted 'Tensor'
        
        Example:
            >>> a = Tensor(np.random.randn(2, 3, 4))
            >>> b = Tensor(np.random.randn(4, 5))
            >>> c = a.contract(b, ([2], [0]))  # Contract last axis of a with first of b 
        """
        
        return Tensor(np.tensordot(self.data, other.data, axes=axes))

    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """
        Reshape the tensor do desired shape.

        :param new_shape: New shape tuple

        Returns:
            Reshaped tensor
        """
        return Tensor(self.data.reshape(new_shape),
                      physical_indices=self.physical_indices.copy(),
                      bond_indices=self.bond_indices.copy()
                      )
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        """
        Transpose the tensor by permuting axes.
        
        Args:
            axes: Permutation of axes. If None, reverses axes.
            
        Returns:
            Transposed tensor.
        """
        return Tensor(np.transpose(self.data, axes), 
                      physical_indices=self.physical_indices.copy(),
                      bond_indices=self.bond_indices.copy()
                      )
    
    def qr_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple['Tensor', 'Tensor']:
        """
        Performs a QR decomposition on the data of the underlying tensor of other strcutures.
        
        Arguments:
            left_indices:
            right_indices:
            
        Returns:
            Tuple of (Q, R) tensors
            
        Algorithm:
            1. Group and reshape into matrix
            2. Perfor QR decomposition
            3. Reshape back to tensor form
        """
        
        # Verify indices:
        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim))
        
        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)
        
        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))
        
        mat = data_perm.reshape(left_dim, right_dim)
        
        Q, R = qr(mat, mode="full")
        
        chi = Q.shape[1]
        Q_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        R_shape = (chi,) + tuple([self.shape[i] for i in right_indices])
        
        # Robustness addition:
        Q_tensor = Tensor(Q.reshape(Q_shape), 
                          physical_indices=self.physical_indices.copy(), 
                          bond_indices=self.bond_indices.copy())
        R_tensor = Tensor(R.reshape(R_shape))
        
        return Q_tensor, R_tensor
    
    def svd_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple['Tensor', 'Tensor', 'Tensor']:
        """
        This is the full SVD decomposition of a tensor without any sort of truncation.
        
        Arguments:
            left_indeces: Indices to group into left matrices
            right_indices: Indices to group into right matrices
            
        Returns:
            Tuple of U, S and V (dagger)
            
        Algorithm:
            1. Combine left/right indices into two groups
            2. Reshape into matrix
            3. SVD
            4. Reshape back to tensor form        
        """
    
        # Verify that all indices are accounted for:
        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim)), "All indices must be specified."
        
        # First we transpose to group left and right indices
        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)
        
        # Calculate dimensions
        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))
        
        # Reshape into matrix
        mat = data_perm.reshape(left_dim, right_dim)
        
        # Perform SVD
        U, S, Vd = svd(mat, full_matrices=False, lapack_driver='gesdd')
        
        # Reshape
        chi = len(S)
        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])
        U_reshaped = Tensor(U.reshape(left_shape))
        V_reshaped = Tensor(Vd.reshape(right_shape))
        
        return U_reshaped, S, V_reshaped
        
    def svd(self, left_indices: List[int], right_indices: List[int], policy: TruncationPolicy | None = None) -> Tuple():
        """
        SVD decomposition of a tensor with optional policy

        """    
    
        U, S, Vt = self.svd_decomposition(left_indices, right_indices)

        if policy is None:
            return U, S, Vt
        
        chi = policy.choose_bond_dim(S)

        U_trunc = Tensor(U.data[..., :chi])
        S_trunc = S[:chi]
        V_trunc = Tensor(Vt.data[:chi, ...])

        return U_trunc, S_trunc, V_trunc
    
    def norm(self) -> float:
        """Returns the Frobenius norm."""
        return float(np.linalg.norm(self.data))
    
    def normalize(self) -> 'Tensor':
        """Returns the normalized tensor."""
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero-norm tensor.")
        return Tensor(self.data / n, 
                      physical_indices=self.physical_indices.copy(),
                      bond_indices=self.bond_indices.copy())

    def __mul__(self, scalar: complex) -> 'Tensor':
        """left scalar multiplication"""
        return Tensor(self.data * scalar)
    
    def __rmul__(self, scalar: complex) -> 'Tensor':
        """Right scalar multiplication"""
        return self.__mul__(scalar)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    def einsum(self, subscripts: str, *others: "Tensor") -> "Tensor":
        arrays = [self.data] + [t.data for t in others]
        out = np.einsum(subscripts, *arrays)
        return Tensor(out)
         
    