from __future__ import annotations  # Postpone evaluation of type hints

from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd, qr, eigh  # NOTE: eigh currently unused, keep or remove
from .policy import TruncationPolicy
from .index import Index
ComplexArray = NDArray[np.complex128]


class Tensor:
    """
    A multi dimensional array wrapper with tensor network operations.

    Attributes:
        data (np.ndarray): The underlying numpy array.
        shape (tuple): Shape/dimensions of the tensor.
        ndim (int): Number of dimensions.
    """

    def __init__(
        self,
        data: ComplexArray | None,
        indices: List[Index] | None = None,
        physical_indices: Optional[List[str]] = None,
        bond_indices: Optional[List[str]] = None,
    ):
        """
        Initialize a tensor from a numpy array.

        Args:
            data: NumPy array
            inds: List of Index objects (one per axis). If None, dummy indices are created.
            physical_indices: Names of physical indices (e.g., ['p_1', 'p_2'])
            bond_indices: Names of bond indices (e.g., ['L', 'R'])
        """
        # Keep dtype as-provided (important: SVD singular values are real floats)
        self.data = None if data is None else np.asarray(data)

        # Initialize indices: 
            # either provided or auto-generated
        if indices is None:
            indices = [Index(dim = d, name = f"axis_{i}") for i, d in enumerate(self.shape)]
        else:
            assert len(indices) == self.ndim, f"Expected {self.ndim} indices, got {len(indices)}."
            
            for ind, d in zip(indices, self.shape):
                assert ind.dim == d, f"Index dim {ind.dim} doesn't match axis dim {d}."

        self.indices = indices

        # keep old metadata for compatibility (depricated, will remove after transition)
        self.physical_indices = physical_indices or []
        self.bond_indices = bond_indices or []

    # ----------------------------
    # Python / NumPy interoperability
    # ----------------------------

    def __len__(self) -> int:
        # Lets Python call len(Tensor) and enables tests like len(S_trunc)
        return len(self.data)

    def __getitem__(self, key):
        # Keep scalar indexing returning scalar; array indexing returns Tensor.
        out = self.data[key]
        if np.isscalar(out) or getattr(out, "shape", ()) == ():
            return out
        return Tensor(out, indices = self.indices.copy())


    def __iter__(self):
        return iter(self.data)

    def __array__(self, dtype=None):
        # Allows np.asarray(Tensor), np.diag(Tensor), np.allclose(Tensor, ...)
        return np.asarray(self.data, dtype=dtype) if dtype is not None else np.asarray(self.data)

    # ----------------------------
    # Minimal arithmetic (add as-needed)
    # ----------------------------

    def __pow__(self, power, modulo=None) -> "Tensor":
        if modulo is not None:
            return NotImplemented
        return Tensor(
            np.power(self.data, power),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def __ge__(self, other) -> "Tensor":
        other_arr = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_arr,
                      indices = self.indices.copy())

    def __mul__(self, scalar: complex) -> "Tensor":
        return Tensor(self.data * scalar,
                      indices=self.indices.copy())

    def __rmul__(self, scalar: complex) -> "Tensor":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        inds_repr = ", ".join(str(i) for i in self.indices)
        return f"Tensor(shape={self.shape}, inds=[{inds_repr}])"

    # ----------------------------
    # Basic properties
    # ----------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.data.ndim

    # ----------------------------
    # Basic tensor ops
    # ----------------------------

    def copy(self) -> "Tensor":
        """Create a deep copy of the tensor."""
        return Tensor(self.data.copy(), 
                      indices = [ind.sim() for ind in self.indices])

    def conj(self) -> "Tensor":
        """Return the complex conjugate of the tensor."""
        return Tensor(np.conj(self.data),
                      indices=self.indices.copy())

    def contract(self, other: "Tensor", axes: Tuple[List[int], List[int]]) -> "Tensor":
        """
        Contract two tensors along specified axis.

        Args:
            other: Tensor to contract with.
            axes: Tuple (self_axes, other_axes) specifying which axes to contract.

        Returns:
            Contracted Tensor with remaining indices.
        """
        # Numeric contraction
        result_data = np.tensordot(self.data, other.data, axes = axes)

        self_remaining = [i for i in range(self.ndim) if i not in axes[0]]
        other_remaining = [i for i in range(other.ndim) if i not in axes[1]]

        result_inds = ([self.indices[i] for i in self_remaining] + [other.indices[i] for i in other_remaining])
        return Tensor(result_data, indices = result_inds)

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        """
        Reshape the tensor to desired shape.
        
        NOTE: Reshape can change index structure; preserve old indices or create new ones.
        For now, we create fresh dummy indices.
        
        Args:
            new_shape: New shape tuple.
        
        Returns:
            Reshaped Tensor.
        """
        new_data = self.data.reshape(new_shape)
        # Create fresh indices (reshape changes structure, so old indices may not apply)
        new_inds = [Index(dim=d, name=f"reshaped_{i}") for i, d in enumerate(new_shape)]
        return Tensor(new_data, indices=new_inds)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        """
        Transpose the tensor by permuting axes.
        
        Args:
            axes: Permutation of axes. If None, reverses axes.
        
        Returns:
            Transposed Tensor.
        """
        new_data = np.transpose(self.data, axes)
        new_inds = [self.indices[i] for i in axes] if axes is not None else list(reversed(self.indices))
        return Tensor(new_data, indices=new_inds)

    def permute_by_inds(self, target_inds: List[Index]) -> "Tensor":
        """
        Permute this tensor to match a target index order.
        
        Finds a permutation of current axes such that indices match target_inds.
        Useful for placing a tensor into a canonical form.
        
        Args:
            target_inds: Desired Index order.
        
        Returns:
            Permuted Tensor matching target order.
        
        Raises:
            ValueError: If current indices don't match target (by id/tags/prime).
        """
        assert len(target_inds) == self.ndim, "Target indices must match tensor rank."
        
        # Find permutation: target_inds[i] should come from self.indices[perm[i]]
        perm = []
        used = set()
        for target_ind in target_inds:
            found = False
            for j, self_ind in enumerate(self.indices):
                if j not in used and self_ind == target_ind:
                    perm.append(j)
                    used.add(j)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Could not find index {target_ind} in tensor indices {self.indices}"
                )
        
        # Apply permutation
        permuted_data = np.transpose(self.data, perm)
        permuted_inds = [self.indices[i] for i in perm]
        return Tensor(permuted_data, indices=permuted_inds)
    # ----------------------------
    # Factorizations
    # ----------------------------

    def qr_decomposition(
        self, left_indices: List[int], right_indices: List[int]
    ) -> Tuple["Tensor", "Tensor"]:
        """
        QR decomposition grouped into (left_indices | right_indices).
        
        Creates a new bond Index for the output.
        
        Returns:
            (Q, R) as Tensors.
        """
        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim))

        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)

        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))

        mat = data_perm.reshape(left_dim, right_dim)
        Q, R = qr(mat, mode="full")

        chi = Q.shape[1]
        
        # Create new bond index
        bond_ind = Index(dim=chi, name="QR_bond", tags={"QR"})
        
        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])
        
        # Gather indices in canonical order: left_inds + bond, then bond + right_inds
        Q_inds = [self.indices[i] for i in left_indices] + [bond_ind]
        R_inds = [bond_ind] + [self.indices[i] for i in right_indices]

        Q_tensor = Tensor(Q.reshape(left_shape), indices=Q_inds)
        R_tensor = Tensor(R.reshape(right_shape), indices=R_inds)

        return Q_tensor, R_tensor

    def svd_decomposition(
        self, left_indices: List[int], right_indices: List[int]
    ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """
        Full SVD decomposition without truncation.
        
        Creates a new bond Index for the output.
        
        Returns:
            (U, S, Vh) as Tensors, where S is 1D.
        """
        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim)), "All indices must be specified."

        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)

        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))

        mat = data_perm.reshape(left_dim, right_dim)
        U, S, Vh = svd(mat, full_matrices=False, lapack_driver="gesdd")

        chi = len(S)
        
        # Create new bond index
        bond_ind = Index(dim=chi, name="SVD_bond", tags={"SVD"})
        
        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])
        
        # Gather indices in canonical order
        U_inds = [self.indices[i] for i in left_indices] + [bond_ind]
        S_inds = [bond_ind]
        Vh_inds = [bond_ind] + [self.indices[i] for i in right_indices]

        U_reshaped = Tensor(U.reshape(left_shape), indices=U_inds)
        V_reshaped = Tensor(Vh.reshape(right_shape), indices=Vh_inds)
        S_tensor = Tensor(S.reshape((chi,)), indices=S_inds)

        return U_reshaped, S_tensor, V_reshaped

    def svd(
        self,
        left_indices: List[int],
        right_indices: List[int],
        policy: Optional[TruncationPolicy] = None,
    ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """
        SVD decomposition with optional truncation policy.
        
        If policy is None, returns full decomposition.
        If policy is provided, truncates along the bond dimension.
        """
        U, S, Vt = self.svd_decomposition(left_indices, right_indices)

        if policy is None:
            return U, S, Vt

        # Singular values are real; cast to float for policy
        s_vals = np.asarray(S.data, dtype=float)
        chi = policy.choose_bond_dim(s_vals)

        # Truncate and preserve indices
        U_trunc = Tensor(
            U.data[..., :chi],
            indices=U.indices[:-1] + [Index(dim=chi, name=U.indices[-1].name, tags=U.indices[-1].tags)],
        )
        S_trunc = Tensor(
            S.data[:chi],
            indices=[Index(dim=chi, name=S.indices[0].name, tags=S.indices[0].tags)],
        )
        V_trunc = Tensor(
            Vt.data[:chi, ...],
            indices=[Index(dim=chi, name=Vt.indices[0].name, tags=Vt.indices[0].tags)] + Vt.indices[1:],
        )

        return U_trunc, S_trunc, V_trunc

    # ----------------------------
    # Norm utilities
    # ----------------------------

    def norm(self) -> float:
        """Returns the Frobenius norm."""
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Tensor":
        """Returns the normalized tensor."""
        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero-norm tensor.")
        return Tensor(
            self.data / n,
            indices=self.indices.copy(),
        )

    # ----------------------------
    # Convenience
    # ----------------------------

    def einsum(self, subscripts: str, *others: "Tensor") -> "Tensor":
        arrays = [self.data] + [t.data for t in others]
        out = np.einsum(subscripts, *arrays)
        return Tensor(out)
