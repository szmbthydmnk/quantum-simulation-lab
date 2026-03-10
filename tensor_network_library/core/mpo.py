"""
Matrix Product Operator (MPO) implementation.

Canonical axis ordering per site tensor:
    axis 0: left bond index
    axis 1: physical in index (row of matrix)
    axis 2: physical out index (column of matrix)
    axis 3: right bond index

Typically the MPO represents the Hamiltonians for DMRG or TEBD.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Callable

import numpy as np
from numpy.typing import NDArray

from .tensor import Tensor
from .index import Index
from .mps import MPS

ComplexArray = NDArray[np.complex128]

class MPO:
    """
    Matrix Product Operator representation.
    
    An MPO represents an operator as a chain of tensors:
    O = Σ W[0]^{s0,s0'} W[1]^{s1,s1'} ... W[L-1]^{s(L-1),s(L-1)'}
    
    Attributes:
        L (int): Chain length.
        d (int): Physical dimension (per site).
        tensors (List[Tensor]): List of MPO tensors. Each has shape (χ_left, d_out, d_in, χ_right).
        bond_dims (list[int]): Bond dimensions [d_0, d_1, d_2, ..., d_L]
        dtype (np.dtype): Data type of tensors.
    """
    
    def __init__(self,
                 L: int, 
                 d: int,
                 bond_policy: Union[str, int, List[int]] = "default",
                 dtype: np.dtype = np.complex128,
                 ):
        """
        Initialize MPO from list of tensors.
        
        Args:
            L: Chain lengths
            d: Physical dimension
            bond_policy:
                - "deafult": [1, 2, 2, ..., 2, 1]
                - int: uniform bond dimension
                - List[int]: explicit bond dimensions (length L + 1)
            dtype: Data type
        """
        
        self.L = L
        self.d = d
        self.dtype = dtype

        # parse bond policy
        if isinstance(bond_policy, list):
            if len(bond_policy) != L + 1:
                raise ValueError(f"bond_policy list must have length L + 1 = {L + 1}, got {len(bond_policy)}")
            self.bond_dims = bond_policy
        elif isinstance(bond_policy, int):
            self.bond_dims = [1] + [bond_policy] * (L - 1) + [1]
        elif bond_policy == "default":
            self.bond_dims = [1] + [2] * (L - 1) + [1]
        else:
            raise ValueError(f"Unknown bond policy: {bond_policy}")


        self.tensors: List[Tensor] = []

        
        # Start with identity initialization, later these will be overwritten with proper operators
        for i in range(L):
            D_left, D_right = self.bond_dims[i], self.bond_dims[i + 1]
            shape = (D_left, d, d, D_right)
            data = np.zeros(shape, dtype = dtype)

            # Identity operators
            if D_left >= 1 and D_right >= 1:
                data[0, :, :, 0] = np.eye(d, dtype = dtype)
            
            # Create indices
            left_ind = Index(dim = D_left, name = f"OL{i}", tags = {f"site_{i}"})
            physical_in = Index(dim = d, name = f"OPI{i}", tags = {f"site_{i}", "physical_in"})
            physical_out = Index(dim = d, name = f"OPO{i}", tags = {f"site_{i}", "physical_out"})
            right_ind = Index(dim = D_right, name = f"OR{i}", tags = {f"site_{i}"})

            self.tensors.append(
                Tensor(data, indices = [left_ind, physical_in, physical_out, right_ind])
                )
  

    # ------------------------
    # Factory Methods
    # ------------------------

    @staticmethod
    def identity_mpo(L: int,
                     d: int, 
                     dtype: np.dtype = np.complex128,
                     ) -> MPO:
        """
        Creates an inditity MPO (trivial, do nothing operator)

        Argumetns: 
            L (int): Length of the chain.
            d (int): physical indices.
            dtype: Data type.
        """

        mpo = MPO(L, d, bond_policy = 1, dtype = dtype)
        for i in range(L):
            mpo.tensors[i].data[:, :, :, :] = 0.0
            mpo.tensors[i].data[0, :, :, 0] = np.eye(d, dtype = dtype)
        
        return mpo
    

    def initialize_random(self) -> None:
        """
        Fill all MPO tensors with random data (presering structure).
        """

        for i in range(self.L):
            shape = self.tensors[i].shape
            data = np.random.randn(*shape).astype(self.dtype)
            if np.issubdtype(self.dtype, np.complexfloating):
                data = data + 1j * np.random.randn(*shape).astype(self.dtype)
            self.tensors[i].data = data

    
    def initialize_single_site_operator(self,
                                        operator: np.ndarray,
                                        site: int,
                                        ) -> None:
        """
        Initialize an MPO as single-site operator at given site.

        Args: 
            operator: (d x d) operator matrix
            site: Site index here operator acts
        """

        if operator.shape != (self.d, self.d):
            raise ValueError(
                f"Operator shape {operator.shape} doesn't match physical dimension ({self.d}, {self.d})"
            )
        
        if not 0 <= site < self.L:
            raise ValueError(f"Site {site} out of range [0, {self.L})")
        
        # Replace at target site, leave the rest of the sites as they were.
        self.tensors[site].data[0, :, :, 0] = operator.astype(self.dtype)


    # -------------------------
    # Python protocol
    # -------------------------

    def __len__(self) -> int:
        return self.L
    

    def __repr__(self) -> str:
        shapes_str = ", ".join([str(t.shape) for t in self.tensors])
        return f"MPO(L={self.L}, d={self.d}, shapes=[{shapes_str}])"
    

    def __str__(self) -> str:
        return self.__repr__()
    

    def __getitem__(self, idx: int) -> Tensor:
        return self.tensors[idx]
    

    # -------------------------
    # Core Operations
    # -------------------------

    def copy(self) -> 'MPO':
        """Creates a deep copy of an MPO."""
        new_mpo = MPO(self.L, self.d, bond_policy=self.bond_dims.copy(), dtype = self.dtype)
        new_mpo.tensors = [t.copy() for t in self.tensors]
        return new_mpo
    
    
    def apply(self, mps: MPS) -> 'MPS':
        """
        Apply this MPO to an MPS: returns a new MPS representing: hat{O} ket{Psi}
        
        The result has a "super-bond" dimension, the product of the MPS and MPO.
        Requires identical physical dimensions.

        Args:
            mps: Input MPS

        Returns:
            New MPS representing O|psi>
        """

        if len(self) != len(mps):
            raise ValueError(
                f"Length mismatch: MPO has {len(self)} sites, while the MPS has {len(mps)} sites."
            )
        
        if self.physical_dims != mps.physical_dims:
            raise ValueError(
                f"Physical dimension mismatch: MPO has {self.physical_dims}, "
                f"MPS has {mps.physical_dims}."
            )
        
        new_tensors = []
        
        # Using Schölwock's notation here for readability:
        for i, (W, M) in enumerate(zip(self.tensors, mps.tensors)):
            w_left, d_in, d_out, w_right = W.shape
            m_left, d_mps, m_right = M.shape
            
            if d_in != d_mps:
                raise ValueError(
                    "Local physical dimension mismatch between MPO ({d_in}) and MPS ({d_mps})"
                    )
            
            # Contract physical indices: W[w_l, d_in, d_out, w_r] * M[m_l, d_in, m_r]
            # axes=([1], [1]) contracts d_in of W with d_mps of M
            temp = np.tensordot(W.data, M.data, axes=([1], [1]))
            # Result shape: (w_left, d_out, w_right, m_left, m_right)

            # Reorder to: (w_left, m_left, d_out, w_right, m_right)
            temp = np.transpose(temp, (0, 3, 1, 2, 4))

            # Merge bond dimensions into "super-bonds":
            m_super_left = w_left * m_left
            m_super_right = w_right * m_right

            new_data = temp.reshape(m_super_left, d_out, m_super_right)

            # creating the proper indices for the results
            left_super_index = Index(
                dim=m_super_left, name=f"super_L{i}", tags={f"site_{i}", "left"}
            )
            right_super_index = Index(
                dim=m_super_right, name=f"super_R{i}", tags={f"site_{i}", "right"}
            )

            # Reuse physical out index from MPO
            physical_index = W.indices[2]

            new_tensors.append(Tensor(
                new_data, indices = [left_super_index, physical_index, right_super_index]
            ))
    
        return MPS.from_tensors(new_tensors, name="MPO_applied")

    

    def to_dense(self) -> np.ndarray:
        """
        Convert the MPO to a full operator matrix O of shape (d^L, d^L).

        Contracts all internal bond indices and arranges physical indices
        in the order (s0_in,...,s(L-1)_in, s0_out,...,s(L-1)_out), then reshapes.

        Intended for small L for debugging/testing only.

        Returns:
            Dense operator matrix of shape (d^L, d^L)
        """
        N = self.L
        d = self.d

        # Start from first tensor: W0 (wL0, d_in0, d_out0, wR0)
        op = self.tensors[0].data

        # Sequentially contract bond indices across sites
        for t in self.tensors[1:]:
            W_next = t.data  # (wL_next, d_in_next, d_out_next, wR_next)

            # Contract right bond of op with left bond of next tensor
            # op: (..., wR), W_next: (wL_next, d_in_next, d_out_next, wR_next)
            op = np.tensordot(op, W_next, axes=([-1], [0]))

        # Remove singleton boundary bond dimensions
        op = np.squeeze(op)

        # Now op has shape (d_in0, d_out0, d_in1, d_out1, ..., d_in(N-1), d_out(N-1))
        # Build permutation: [d_in0,...,d_in(N-1), d_out0,...,d_out(N-1)]
        axes = list(range(op.ndim))
        in_axes = axes[0::2]   # [0, 2, 4, ...]
        out_axes = axes[1::2]  # [1, 3, 5, ...]
        perm = in_axes + out_axes

        op_perm = np.transpose(op, perm)

        dim = d ** N
        return op_perm.reshape(dim, dim)
    

    # -------------------------
    # Basic properties
    # -------------------------

    @property
    def shape(self) -> List[Tuple[int, int, int, int]]:
        """ Returns the shape of all MPO tensors as a list of tuples."""
        return [t.shape for t in self.tensors]


    @property
    def physical_dims(self) -> List[int]:
        """Physical dimensions of the sites."""
        return [t.shape[1] for t in self.tensors]
    
    
    

    
        