"""Matrix Product Operator (MPO) implementation."""

import numpy as np
from typing import List
from .tensor import Tensor

class MPO:
    """
    Matrix Product Operator representation.
    
    An MPO represents an operator as a chain of tensors:
    O = Σ W[0]^{s0,s0'} W[1]^{s1,s1'} ... W[L-1]^{s(L-1),s(L-1)'}
    
    Attributes:
        tensors (List[Tensor]): List of MPO tensors. Each has shape (χ_left, d_out, d_in, χ_right).
        num_sites (int): Number of sites.
    """
    
    def __init__(self, tensors: List[Tensor]):
        """
        Initialize MPO from list of tensors.
        
        Args:
            tensors: List of MPO tensors with shape (chi_left, d_out, d_in, chi_right)
        """
        self.tensors = tensors
        self.num_sites = len(tensors)
        
        # Validate bond dimension
        for i in range(self.num_sites - 1):
            if tensors[i].shape[3] != tensors[i + 1].shape[0]:
                raise ValueError(f"Bond dimension mismatch at site {i}: "
                                 f"{tensors[i].shape[3]} != {tensors[i + 1].shape[0]}")
                
    def __len__(self) -> int:
        return self.num_sites
    
    def copy(self) -> 'MPO':
        """Creates a deep copy of an MPO."""
        return MPO([t.copy() for t in self.tensors])
    
    @property
    def bond_dims(self) ->List[int]:
        """Bond dimensions between sites (incuding boundaires)."""
        dims = [self.tensors[0].shape[0]] # this is the left boundary
        for t in self.tensors:
            dims.append(t.shape[3])
        return dims

    @property
    def physical_dims(self) -> List[int]:
        """Physical dimensions of the sites."""
        return [t.shape[1] for t in self.tensors]
    
    @classmethod
    def identity(cls, num_sites: int, physical_dims: int = 2) -> 'MPO':
        """
        Identity operator MPO: acts as the identity operator on each local Hilbert space.
        
        Each tensor has shape (1, d, d, 1) with identity on the (d, d) block.
        """
        
        tensors: List[Tensor] = []
        
        for _ in range(num_sites):
            data = np.zeros((1, physical_dims, physical_dims, 1), dtype = np.complex128)
            for i in range(physical_dims):
                data[0, i, i, 0] = 1.0
            
            tensors.append(Tensor(data))
        
        return cls(tensors)