from __future__ import annotations  # Postpone evaluation of type hints (allows forward references, avoids circular imports)


from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd, qr, eigh
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
        return Tensor(self.data.copy())
    
    def conj(self) -> 'Tensor':
        """Return the complex conugate of the tensor"""
        return Tensor(np.conj(self.data))
    
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

    def einsum(self, subscripts: str, *others: "Tensor") -> "Tensor":
        arrays = [self.data] + [t.data for t in others]
        out = np.einsum(subscripts, *arrays)
        return Tensor(out)
         
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"