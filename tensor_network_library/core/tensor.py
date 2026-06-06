"""Core Tensor class with Index-based contraction."""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from .index import Index
from .policy import TruncationPolicy


class Tensor:
    """
    A named, multi-dimensional array whose axes are labelled by Index objects.

    Parameters
    ----------
    data :
        The underlying array, or ``None`` for a structure-only tensor.
    indices :
        Ordered list of :class:`Index` objects, one per axis.
    name :
        Human-readable label (optional).
    """

    def __init__(
        self,
        data: Optional[np.ndarray],
        indices: List[Index],
        name: str = "",
    ) -> None:
        self.data: Optional[np.ndarray] = (
            np.asarray(data) if data is not None else None
        )
        self.indices: List[Index] = list(indices)
        self.name: str = str(name)

        if self.data is not None:
            self._check_shape()

    # ------------------------------------------------------------------
    # Shape / structure
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """Number of axes."""
        return len(self.indices)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape derived from Index objects."""
        return tuple(ix.dim for ix in self.indices)

    @property
    def ndim(self) -> int:
        return len(self.indices)

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def _check_shape(self) -> None:
        if self.data is None:
            return
        expected = tuple(ix.dim for ix in self.indices)
        if self.data.shape != expected:
            raise ValueError(
                f"Data shape {self.data.shape} does not match "
                f"index dims {expected}."
            )

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def materialize_zeros(
        self,
        dtype: np.dtype = np.complex128,
    ) -> None:
        """Fill with zeros (in-place)."""
        self.data = np.zeros(self.shape, dtype=dtype)

    def materialize_random(
        self,
        seed: Optional[int] = None,
        dtype: np.dtype = np.complex128,
    ) -> None:
        """Fill with random normal data (in-place)."""
        rng = np.random.default_rng(seed)
        self.data = rng.standard_normal(self.shape).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            self.data = self.data + 1j * rng.standard_normal(self.shape).astype(dtype)

    # ------------------------------------------------------------------
    # Arithmetic / norms
    # ------------------------------------------------------------------

    def norm(self) -> float:
        """Frobenius norm of the underlying array."""
        if self.data is None:
            raise RuntimeError("Tensor is not materialized.")
        return float(np.linalg.norm(self.data))

    def normalize(self) -> float:
        """Normalize the tensor in-place; returns the old norm."""
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize a zero tensor.")
        self.data = self.data / n
        return n

    def __mul__(self, scalar) -> "Tensor":
        if self.data is None:
            raise RuntimeError("Tensor is not materialized.")
        return Tensor(self.data * scalar, list(self.indices), name=self.name)

    def __rmul__(self, scalar) -> "Tensor":
        return self.__mul__(scalar)

    def __add__(self, other: "Tensor") -> "Tensor":
        if self.data is None or other.data is None:
            raise RuntimeError("Both tensors must be materialized for addition.")
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Tensor(self.data + other.data, list(self.indices), name=self.name)

    # ------------------------------------------------------------------
    # Contraction
    # ------------------------------------------------------------------

    def contract(self, other: "Tensor") -> "Tensor":
        """
        Contract *self* with *other* over all shared indices.

        Shared indices are identified by object identity (same ``Index`` instance).
        The output tensor contains all non-shared indices, in the order
        (self_free_indices, other_free_indices).
        """
        if self.data is None or other.data is None:
            raise RuntimeError("Both tensors must be materialized to contract.")

        self_shared  = [i for i, ix in enumerate(self.indices)  if ix in other.indices]
        other_shared = [other.indices.index(self.indices[s]) for s in self_shared]

        self_free  = [i for i in range(self.rank)  if i not in self_shared]
        other_free = [i for i in range(other.rank) if i not in other_shared]

        result_data = np.tensordot(self.data, other.data, axes=(self_shared, other_shared))

        result_indices = (
            [self.indices[i]  for i in self_free] +
            [other.indices[i] for i in other_free]
        )

        return Tensor(result_data, result_indices)

    # ------------------------------------------------------------------
    # SVD
    # ------------------------------------------------------------------

    def svd_decomposition(
        self,
        left_indices: List[Index],
        *,
        truncation: Optional[TruncationPolicy] = None,
        absorb: str = "right",
    ) -> Tuple["Tensor", np.ndarray, "Tensor", Index]:
        """
        SVD of this tensor, splitting axes into left and right groups.

        Parameters
        ----------
        left_indices :
            The subset of ``self.indices`` that form the *left* (U) tensor.
            The remaining indices form the *right* (Vh) tensor.
        truncation :
            Optional truncation policy.  If ``None``, keep all singular values.
        absorb :
            Where to absorb the singular values: ``"right"`` (default),
            ``"left"``, or ``"sqrt"`` (split symmetrically).

        Returns
        -------
        U : Tensor
            Left unitary, indices = left_indices + [new_bond].
        S : np.ndarray
            1-D array of (kept) singular values.
        Vh : Tensor
            Right tensor, indices = [new_bond] + right_indices.
        bond : Index
            The new shared bond Index.
        """
        if self.data is None:
            raise RuntimeError("Tensor is not materialized.")

        # Determine right indices
        right_indices = [ix for ix in self.indices if ix not in left_indices]

        # Transpose data so left axes come first
        left_pos  = [self.indices.index(ix) for ix in left_indices]
        right_pos = [self.indices.index(ix) for ix in right_indices]
        perm = left_pos + right_pos
        data = np.transpose(self.data, perm)

        rows = int(np.prod([self.indices[p].dim for p in left_pos]))
        cols = int(np.prod([self.indices[p].dim for p in right_pos]))
        mat  = data.reshape(rows, cols)

        U_mat, S, Vh_mat = np.linalg.svd(mat, full_matrices=False)

        # Truncate
        if truncation is not None:
            k = truncation.choose_bond_dim(S)
            U_mat, S, Vh_mat = U_mat[:, :k], S[:k], Vh_mat[:k, :]
        else:
            k = len(S)

        # New shared bond index
        bond = Index(dim=k, name=f"{self.name}_svd_bond", tags=frozenset({"svd_bond"}))

        # Absorb singular values
        if absorb == "right":
            U_data  = U_mat
            Vh_data = np.diag(S) @ Vh_mat
        elif absorb == "left":
            U_data  = U_mat * S[np.newaxis, :]
            Vh_data = Vh_mat
        elif absorb == "sqrt":
            sqS     = np.sqrt(S)
            U_data  = U_mat * sqS[np.newaxis, :]
            Vh_data = np.diag(sqS) @ Vh_mat
        else:
            raise ValueError(f"Unknown absorb mode: {absorb!r}")

        # Reshape back
        left_shape  = [self.indices[p].dim for p in left_pos]  + [k]
        right_shape = [k] + [self.indices[p].dim for p in right_pos]

        U_tensor  = Tensor(U_data.reshape(left_shape),   left_indices  + [bond])
        Vh_tensor = Tensor(Vh_data.reshape(right_shape), [bond] + right_indices)

        return U_tensor, S, Vh_tensor, bond


    def svd(
        self,
        left_indices: List[Index],
        *,
        truncation: Optional[TruncationPolicy] = None,
        absorb: str = "right",
    ) -> Tuple["Tensor", np.ndarray, "Tensor", Index]:
        """Alias for :meth:`svd_decomposition`."""
        return self.svd_decomposition(
            left_indices, truncation=truncation, absorb=absorb
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> "Tensor":
        """Deep copy."""
        new_data = self.data.copy() if self.data is not None else None
        return Tensor(new_data, list(self.indices), name=self.name)

    def __repr__(self) -> str:
        shape = self.shape
        mat   = "yes" if self.data is not None else "no"
        return f"Tensor(shape={shape}, materialized={mat}, name={self.name!r})"
