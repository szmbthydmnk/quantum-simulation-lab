"""Matrix Product Operator (MPO) implementation."""

import numpy as np
from typing import List
from .tensor import Tensor
from .mps import MPS

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
    
    def apply(self, mps: MPS) -> 'MPS':
        """
        Apply this MPO to an MPS: returns a new MPS representing: $$\hat{O}\ket{\Psi}$$
        
        Requires identical physical dimension
        """
        
        if len(self) != len(mps):
            raise ValueError(f"Length mismatch: MPO has {len(self)} sites, MPS has {len(mps)} sites")
        
        if self.physical_dims != mps.physical_dims:
            raise ValueError(f"Physical dimension mismatch: MPO has physical dimension {self.physical_dims} while the MPS has {mps.physical_dims} physical dimension.")
        
        new_tensors = []
        
        # Using Schölwock's notation here for readability:
        for W, M in zip(self.tensors, mps.tensors):
            w_left, d_in, d_out, w_right = W.shape
            m_left, d_mps, m_right = M.shape
            
            if d_in != d_mps:
                raise ValueError("Local physical dimension mismatch between MPO ({d_in}) and MPS ({d_mps})")
            
            temp = np.tensordot(W.data, M.data, axes = ([1], [1]))  # The shape here is (w_left, d_out, w_right, m_left, m_right)
            
            temp = np.transpose(temp,(0, 3, 1, 2, 4))   # Reorder the indices to match the convention (w_left, m_left, d_out, w_right, m_right)
            
            # Contract w_left, m_left and w_right, m_right -> "Superindices"
            m_super_left = w_left * m_left
            m_super_right = w_right * m_right
            
            new_data = temp.reshape(m_super_left, d_out, m_super_right)
            
            new_tensors.append(Tensor(new_data))
            
        return MPS(new_tensors)