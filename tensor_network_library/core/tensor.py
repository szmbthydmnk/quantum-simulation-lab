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

    Supports two modes:
        - materialized: data is a NumPy array
        - unmaterialized: data is None, but indices define shape/ndim
    """

    def __init__(
        self,
        data: ComplexArray | None,
        indices: List[Index] | None = None,
        physical_indices: Optional[List[str]] = None,
        bond_indices: Optional[List[str]] = None,
    ):
        """
        Initialize a tensor.

        Args:
            data: NumPy array or None (unmaterialized).
            indices: List[Index] (one per axis). Required if data is None.
            physical_indices: Deprecated string metadata, kept for compatibility.
            bond_indices: Deprecated string metadata, kept for compatibility.
        """
        self.data = None if data is None else np.asarray(data)

        # keep old metadata for compatibility (deprecated, will remove after transition)
        self.physical_indices = physical_indices or []
        self.bond_indices = bond_indices or []

        # ----------------------------
        # Unmaterialized tensor path
        # ----------------------------
        if self.data is None:
            if indices is None:
                raise ValueError("Unmaterialized Tensor requires explicit indices.")
            self.indices = list(indices)
            return

        # ----------------------------
        # Materialized tensor path
        # ----------------------------
        if indices is None:
            self.indices = [Index(dim=d, name=f"axis_{i}") for i, d in enumerate(self.data.shape)]
        else:
            assert len(indices) == self.data.ndim, f"Expected {self.data.ndim} indices, got {len(indices)}."
            for ind, d in zip(indices, self.data.shape):
                assert ind.dim == d, f"Index dim {ind.dim} doesn't match axis dim {d}."
            self.indices = list(indices)

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _require_data(self) -> None:
        if self.data is None:
            raise ValueError("Tensor has data=None (unmaterialized). Materialize before numeric ops.")

    def is_materialized(self) -> bool:
        return self.data is not None

    def materialize_zeros(self, dtype: np.dtype = np.complex128) -> "Tensor":
        """
        Materialize an unmaterialized tensor as an explicit dense zeros array.
        """
        if self.data is None:
            self.data = np.zeros(self.shape, dtype=dtype)
        return self

    # ----------------------------
    # Python / NumPy interoperability
    # ----------------------------

    def __len__(self) -> int:
        self._require_data()
        return len(self.data)

    def __getitem__(self, key):
        self._require_data()

        out = self.data[key]
        if np.isscalar(out) or getattr(out, "shape", ()) == ():
            return out

        # Safe default: slicing can change rank/shape, so create fresh indices.
        new_inds = [Index(dim=d, name=f"axis_{i}") for i, d in enumerate(out.shape)]
        return Tensor(out, indices=new_inds)

    def __iter__(self):
        self._require_data()
        return iter(self.data)

    def __array__(self, dtype=None):
        self._require_data()
        return np.asarray(self.data, dtype=dtype) if dtype is not None else np.asarray(self.data)

    # ----------------------------
    # Minimal arithmetic (add as-needed)
    # ----------------------------

    def __pow__(self, power, modulo=None) -> "Tensor":
        self._require_data()
        if modulo is not None:
            return NotImplemented
        return Tensor(np.power(self.data, power), indices=self.indices.copy())

    def __ge__(self, other) -> "Tensor":
        self._require_data()
        other_arr = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_arr, indices=self.indices.copy())

    def __mul__(self, scalar: complex) -> "Tensor":
        self._require_data()
        return Tensor(self.data * scalar, indices=self.indices.copy())

    def __rmul__(self, scalar: complex) -> "Tensor":
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        inds_repr = ", ".join(str(i) for i in self.indices) if hasattr(self, "indices") else ""
        return f"Tensor(shape={self.shape}, inds=[{inds_repr}])"

    # ----------------------------
    # Basic properties
    # ----------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns shape of the tensor."""
        if self.data is not None:
            return self.data.shape
        return tuple(ind.dim for ind in self.indices)

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        if self.data is not None:
            return self.data.ndim
        return len(self.indices)

    # ----------------------------
    # Basic tensor ops
    # ----------------------------

    def copy(self) -> "Tensor":
        """
        Create a deep copy of the tensor.

        Note: by default we preserve indices (do not sim()) so shared connectivity
        inside a tensor network can be preserved when copying the network.
        """
        if self.data is None:
            return Tensor(None, indices=self.indices.copy())
        return Tensor(self.data.copy(), indices=self.indices.copy())

    def conj(self) -> "Tensor":
        self._require_data()
        return Tensor(np.conj(self.data), indices=self.indices.copy())

    def contract(self, other: "Tensor", axes: Tuple[List[int], List[int]]) -> "Tensor":
        """
        Contract two tensors along specified axes.

        Returns:
            Contracted Tensor with remaining indices.
        """
        self._require_data()
        other._require_data()

        result_data = np.tensordot(self.data, other.data, axes=axes)

        self_remaining = [i for i in range(self.ndim) if i not in axes[0]]
        other_remaining = [i for i in range(other.ndim) if i not in axes[1]]

        result_inds = [self.indices[i] for i in self_remaining] + [other.indices[i] for i in other_remaining]
        return Tensor(result_data, indices=result_inds)

    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        self._require_data()

        new_data = self.data.reshape(new_shape)
        new_inds = [Index(dim=d, name=f"reshaped_{i}") for i, d in enumerate(new_shape)]
        return Tensor(new_data, indices=new_inds)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> "Tensor":
        self._require_data()

        new_data = np.transpose(self.data, axes)
        if axes is None:
            new_inds = list(reversed(self.indices))
        else:
            new_inds = [self.indices[i] for i in axes]
        return Tensor(new_data, indices=new_inds)

    def permute_by_inds(self, target_inds: List[Index]) -> "Tensor":
        self._require_data()

        assert len(target_inds) == self.ndim, "Target indices must match tensor rank."

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
                raise ValueError(f"Could not find index {target_ind} in tensor indices {self.indices}")

        permuted_data = np.transpose(self.data, perm)
        permuted_inds = [self.indices[i] for i in perm]
        return Tensor(permuted_data, indices=permuted_inds)

    # ----------------------------
    # Factorizations
    # ----------------------------

    def qr_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple["Tensor", "Tensor"]:
        self._require_data()

        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim))

        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)

        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))

        mat = data_perm.reshape(left_dim, right_dim)
        Q, R = qr(mat, mode="full")

        chi = Q.shape[1]

        bond_ind = Index(dim=chi, name="QR_bond", tags=frozenset({"QR"}))

        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])

        Q_inds = [self.indices[i] for i in left_indices] + [bond_ind]
        R_inds = [bond_ind] + [self.indices[i] for i in right_indices]

        Q_tensor = Tensor(Q.reshape(left_shape), indices=Q_inds)
        R_tensor = Tensor(R.reshape(right_shape), indices=R_inds)
        return Q_tensor, R_tensor

    def svd_decomposition(self, left_indices: List[int], right_indices: List[int]) -> Tuple["Tensor", "Tensor", "Tensor"]:
        self._require_data()

        all_indices = set(left_indices + right_indices)
        assert all_indices == set(range(self.ndim)), "All indices must be specified."

        perm = left_indices + right_indices
        data_perm = np.transpose(self.data, perm)

        left_dim = int(np.prod([self.shape[i] for i in left_indices]))
        right_dim = int(np.prod([self.shape[i] for i in right_indices]))

        mat = data_perm.reshape(left_dim, right_dim)
        U, S, Vh = svd(mat, full_matrices=False, lapack_driver="gesdd")

        chi = len(S)

        bond_ind = Index(dim=chi, name="SVD_bond", tags=frozenset({"SVD"}))

        left_shape = tuple([self.shape[i] for i in left_indices]) + (chi,)
        right_shape = (chi,) + tuple([self.shape[i] for i in right_indices])

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
        self._require_data()

        U, S, Vt = self.svd_decomposition(left_indices, right_indices)

        if policy is None:
            return U, S, Vt

        s_vals = np.asarray(S.data, dtype=float)
        chi = policy.choose_bond_dim(s_vals)

        # Keep the same bond identity (id) and metadata, only change dim.
        old_bond = U.indices[-1]
        new_bond = Index(dim=chi, name=old_bond.name, tags=old_bond.tags, prime=old_bond.prime, id=old_bond.id)

        U_trunc = Tensor(U.data[..., :chi], indices=U.indices[:-1] + [new_bond])
        S_trunc = Tensor(S.data[:chi], indices=[new_bond])
        V_trunc = Tensor(Vt.data[:chi, ...], indices=[new_bond] + Vt.indices[1:])

        return U_trunc, S_trunc, V_trunc

    # ----------------------------
    # Norm utilities
    # ----------------------------

    def norm(self) -> float:
        self._require_data()
        return float(np.linalg.norm(self.data))

    def normalize(self) -> "Tensor":
        self._require_data()

        n = self.norm()
        if n == 0.0:
            raise ValueError("Cannot normalize a zero-norm tensor.")
        return Tensor(self.data / n, indices=self.indices.copy())

    # ----------------------------
    # Convenience
    # ----------------------------

    def einsum(self, subscripts: str, *others: "Tensor") -> "Tensor":
        self._require_data()
        for t in others:
            t._require_data()

        arrays = [self.data] + [t.data for t in others]
        out = np.einsum(subscripts, *arrays)

        # If you want index-aware einsum later, this is where it would go.
        return Tensor(out)
