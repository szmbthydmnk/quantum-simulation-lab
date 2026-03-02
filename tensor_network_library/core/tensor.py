from __future__ import annotations  # Postpone evaluation of type hints

from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd, qr, eigh  # NOTE: eigh currently unused, keep or remove
from .policy import TruncationPolicy

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
        data: ComplexArray,
        physical_indices: Optional[List[str]] = None,
        bond_indices: Optional[List[str]] = None,
    ):
        """
        Initialize a tensor from a numpy array.

        Args:
            data: NumPy array
            physical_indices: Names of physical indices (e.g., ['p_1', 'p_2'])
            bond_indices: Names of bond indices (e.g., ['L', 'R'])
        """
        # Keep dtype as-provided (important: SVD singular values are real floats)
        self.data = np.asarray(data)
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
        return Tensor(
            out,
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

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
        return Tensor(
            self.data >= other_arr,
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def __mul__(self, scalar: complex) -> "Tensor":
        """Left scalar multiplication."""
        return Tensor(
            self.data * scalar,
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def __rmul__(self, scalar: complex) -> "Tensor":
        """Right scalar multiplication."""
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"

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
        return Tensor(
            self.data.copy(),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def conj(self) -> "Tensor":
        """Return the complex conjugate of the tensor."""
        return Tensor(
            np.conj(self.data),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def contract(self, other: "Tensor", axes: Tuple[List[int], List[int]]) -> "Tensor":
        """
        Contract two tensors along specified axis.

        Args:
            other: Tensor to contract with.
            axes: Tuple (self_axes, other_axes) specifying which axes to contract.

        Returns:
            Contracted Tensor.
        """
        # NOTE: proper index bookkeeping for contractions needs per-axis labels;
        # for now we only return numeric result.
        return Tensor(np.tensordot(self.data, other.data, axes=axes))

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        """
        Reshape the tensor to desired shape.

        Args:
            new_shape: New shape tuple.

        Returns:
            Reshaped Tensor.
        """
        return Tensor(
            self.data.reshape(new_shape),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        """
        Transpose the tensor by permuting axes.

        Args:
            axes: Permutation of axes. If None, reverses axes.

        Returns:
            Transposed Tensor.
        """
        return Tensor(
            np.transpose(self.data, axes),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    # ----------------------------
    # Factorizations
    # ----------------------------

    def qr_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple["Tensor", "Tensor"]:
        """
        Performs a QR decomposition of the tensor grouped into (left_indices | right_indices).

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
        Q_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        R_shape = (chi,) + tuple([self.shape[i] for i in right_indices])

        Q_tensor = Tensor(
            Q.reshape(Q_shape),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )
        R_tensor = Tensor(R.reshape(R_shape))

        return Q_tensor, R_tensor

    def svd_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """
        Full SVD decomposition of a tensor without truncation.

        Returns:
            (U, S, Vh) as Tensors, where S is a 1D tensor of singular values.
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
        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])

        U_reshaped = Tensor(
            U.reshape(left_shape),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )
        V_reshaped = Tensor(
            Vh.reshape(right_shape),
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )
        S_tensor = Tensor(
            S,
            physical_indices=[],
            bond_indices=[],
        )

        return U_reshaped, S_tensor, V_reshaped

    def svd(
        self,
        left_indices: List[int],
        right_indices: List[int],
        policy: TruncationPolicy | None = None,
    ) -> Tuple["Tensor", "Tensor", "Tensor"]:
        """
        SVD decomposition of a tensor with optional truncation policy.

        If policy is None:
            Returns full (U, S, Vh).

        If policy is provided:
            Chooses chi via policy and truncates along the bond dimension,
            while preserving metadata from the full decomposition outputs.
        """
        U, S, Vt = self.svd_decomposition(left_indices, right_indices)

        if policy is None:
            return U, S, Vt

        # Singular values are real; cast to float for policy.
        s_vals = np.asarray(S.data, dtype=float)
        chi = policy.choose_bond_dim(s_vals)

        # IMPORTANT: preserve metadata from U/S/Vt instead of dropping it.
        U_trunc = Tensor(
            U.data[..., :chi],
            physical_indices=U.physical_indices.copy(),
            bond_indices=U.bond_indices.copy(),
        )
        S_trunc = Tensor(
            S.data[:chi],
            physical_indices=S.physical_indices.copy(),
            bond_indices=S.bond_indices.copy(),
        )
        V_trunc = Tensor(
            Vt.data[:chi, ...],
            physical_indices=Vt.physical_indices.copy(),
            bond_indices=Vt.bond_indices.copy(),
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
            physical_indices=self.physical_indices.copy(),
            bond_indices=self.bond_indices.copy(),
        )

    # ----------------------------
    # Convenience
    # ----------------------------

    def einsum(self, subscripts: str, *others: "Tensor") -> "Tensor":
        arrays = [self.data] + [t.data for t in others]
        out = np.einsum(subscripts, *arrays)
        return Tensor(out)
