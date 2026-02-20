"""Matrix Product State (MPS) implementation."""

import numpy as np
from typing import List, Optional, Tuple
from .tensor import Tensor

class MPS:
    """
    Matrix Product State (MPS) representation.
    
    An MPS represents a quantum state as a chain of tensors:
    $$\ket{\Psi} = \sum A[1]^{s1} ... A[L-1]^{s(L-1)} \ket{s0,s1,...,s(L-1)}$$
    
    Attributes:
        tensors (List[Tensor]): List of MPS tensors. Each tensor has shape (χ_left, d, χ_right)
                                where d is physical dimension and χ are bond dimensions.
        num_sites (int): Number of sites in the chain.
        physical_dims (List[int]): Physical dimension at each site.
        bond_dims (List[int]): Bond dimensions between sites.
    """
    
    def __init__(self, tensors: List[Tensor]):
        """
        Initilaize an MPS from list of Tensors.
        
        Args:
            tensors: [List[Tensor]]; List of MPS tensors with shape (chi_left, d, chi_right).
        """
        
        self.tensors = tensors
        self.num_sites = len(tensors)
        
        # Validate dimensions:
        for i in range(self.num_sites - 1):
            if tensors[i].shape[2] != tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Bond dimension mismatch at site {i}: "
                    f"{tensors[i].shape[2]} != {tensors[i + 1].shape[0]}"
                )
            
            
            # assert tensors[i].shape[2] != tensors[i + 1].shape[0], f"Bond dimension mismatch at site {i}: {tensors[i].shape[2]} != {tensors[i + 1].shape[0]}"
                
    def __len__(self):
        return len(self.tensors)
    
    @property
    def bond_dims(self) -> List[int]:
        """Bond dimensions between sites (including boundaries)"""
        dims = [self.tensors[0].shape[0]]     # Left boundary
        for t in self.tensors:
            dims.append(t.shape[2])
        return dims
    
    @property
    def physical_dims(self) -> List[int]:
        """Physical dimension at each site"""
        return [t.shape[1] for t in self.tensors]
        
    def norm(self) -> float:
        """
        Compute the norm of the MPS as √⟨ψ|ψ⟩.
        
        Algorithm:
            Contract the MPS with its conjugate from left to right.
        """
        # Start with the identity
        overlap = np.eye(self.tensors[0].shape[0], dtype = np.complex128)
        
        for tensor in self.tensors:
            # Contract: overlap with tensor and its conjugate
            # overlap shape: (χ_left, χ_left')
            # tensor shape: (χ_left, d, χ_right)
            
            # First contract overlap with tensor
            temp = np.tensordot(overlap, tensor.data, axes = ([0], [0]))
            # temp shape: (χ_left', d, χ_right)
            
            # Then contract with conjugate
            temp = np.tensordot(temp, tensor.conj().data, axes = ([0, 1], [0, 1]))
            # temp shape: (χ_right, χ_right')
            
            overlap = temp
        
        # Trace the final matrix
        return np.sqrt(np.abs(np.trace(overlap)))
    
    def normalize(self):
        """
        Normalize the MPS in-place so that ⟨ψ|ψ⟩ = 1.
        
        Returns:
            Self for chaining.
        """
        norm = self.norm()
        if norm > 0:
            # Normalize the first tensor
            self.tensors[0] = self.tensors[0] * (1.0 / norm)
        return self
        
    def copy(self) -> 'MPS':
        """Creates a deep copy of the MPS"""
        return MPS([t.copy() for t in self.tensors])
        
    @classmethod
    def from_product_state(cls, state_indices: List[int], physical_dims: int = 2) -> 'MPS':
        """
        Create an MPS from a product state of computational basis states.

        Args:
            state_indices: List of local state indices (0 to physical_dim-1) for each site.
            physical_dim: Physical dimension (default 2 for qubits).

        Returns:
            MPS representing the product state.

        Example:
            >>> mps = mps_from_product_state([0, 1, 0, 1])  # |0101⟩
        """
        
        num_sites = len(state_indices)
        
        tensors = []
        
        for idx in state_indices:
            
            tensor_data = np.zeros((1, physical_dims, 1), dtype = np.complex128)
            tensor_data[0, idx, 0] = 1.0
            tensors.append(Tensor(tensor_data))
        return cls(tensors)
    
    @classmethod
    def from_local_states(cls, local_states: List[int]) -> 'MPS':
        """
        Creates a product MPS state from local states, not limited to the computational basis.
        
        Args:
            local_states[i]: 
        Returns:
        
        Example(s):
        """
        
    
    def to_dense(self) -> np.ndarray:
        """
        Convert the MPS to a full statevector $$\ket{\Psi}$$ as a 1D array of length $$d^N$$.
        
        This helper function is intended for small systems and mostly for error correction/checks.
        """
        
        # Merge the bonds of the first tensor
            # the shape of the first tensor is M[0]: (chi_left, d, chi_right)
        M0 = self.tensors[0].data
        psi = M0
        
        for t in self.tensors[1:]:
            # t: (mL_next, d, mR_next)
            # contract current right bond with next left bond
            # psi: (mL, d1, mR), t: (mL2, d2, mR2)
            # require mR == mL2
            psi = np.tensordot(psi, t.data, axes = ([psi.ndim - 1], [0]))
            # psi has shape now: (m_left, d_1, d_2, m_right_2) for second step, ...
        
        psi = np.squeeze(psi)   # remove any 1-sized boundary dimensions
        return psi.reshape(-1)