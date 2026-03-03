"""Matrix Product State (MPS) implementation."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from .tensor import Tensor
from .index import Index
from .policy import TruncationPolicy


BondPolicy = Union[str, List[int]]
PhysDims = Union[int, List[int]]


class MPS:
    """
    Matrix Product State with Index-based architecture.

    Structure:
        - Each site i has a tensor with indices [bond_left, physical, bond_right]
        - Bonds connect adjacent tensors via shared Index objects.
        - Supports arbitrary bond dimensions via policy.
    """

    def __init__(
        self,
        L: int,
        physical_dims: PhysDims = 2,
        bond_policy: BondPolicy = "default",
        name: str = "MPS",
        truncation: TruncationPolicy | None = None,
        dtype: np.dtype = np.complex128,
    ):
        """
        Initialize an MPS *structure* (Indices + site tensors).

        Args:
            L: Chain length.
            physical_dims: Physical dimension(s). If int, same for all sites.
                If List, must have length L.
            bond_policy:
                - "default": chi_i = min(prod_{k<=i} d_k, prod_{k>i} d_k), optionally capped by truncation.chi_max.
                - "uniform": uses truncation.chi_max as internal chi (boundaries stay 1).
                - List[int]: explicit bond dims of length L+1, boundaries should be 1.
            name: Name tag for this MPS (used in Index names).
            truncation: Optional truncation policy (used to cap bond dims for "default"/"uniform").
            allocate: If True, allocate tensors with zeros; if False, keep data=None (structure-only).
            dtype: dtype used when allocating.
        """
        if L <= 0:
            raise ValueError("L must be a positive integer")

        self.L = int(L)
        self.name = str(name)

        # These are Index objects (not ints).
        self.indices: List[Index] = []  # physical indices
        self.bonds: List[Index] = []    # bond indices

        # Site tensors (Tensor objects)
        self.tensors: List[Tensor] = []

        # Normalize physical dims to list[int]
        self._physical_dims: List[int] = self._parse_physical_dims(physical_dims)

        # Resolve bond dims (list[int] of length L+1)
        self._bond_dims: List[int] = self._resolve_bond_dims(bond_policy=bond_policy, truncation=truncation)

        # Create Index objects
        self.indices = [
            Index(dim=self._physical_dims[i], name=f"{self.name}_phys_{i}", tags={"phys", f"i={i}"})
            for i in range(self.L)
        ]
        self.bonds = [
            Index(dim=self._bond_dims[i], name=f"{self.name}_bond_{i}", tags={"bond", f"b={i}"})
            for i in range(self.L + 1)
        ]

        # Create tensors
        self._create_empty_tensors(dtype=dtype)

    # -------------------------
    # Constructors / factories
    # -------------------------

    @classmethod
    def from_tensors(cls, tensors: List[Tensor], name: str = "MPS") -> "MPS":
        """
        Create an MPS from a list of already-formed site tensors.

        Assumes each tensor has indices [bond_left, physical, bond_right].
        """
        if len(tensors) == 0:
            raise ValueError("tensors must be a non-empty list")

        obj = cls.__new__(cls)
        obj.name = str(name)
        obj.tensors = [t.copy() for t in tensors]
        obj.L = len(obj.tensors)

        # Extract indices from tensors
        obj.indices = [t.indices[1] for t in obj.tensors]
        obj.bonds = [obj.tensors[0].indices[0]] + [t.indices[2] for t in obj.tensors]

        obj._physical_dims = [ix.dim for ix in obj.indices]
        obj._bond_dims = [ix.dim for ix in obj.bonds]
        return obj

    @classmethod
    def from_product_state(
        cls,
        state_indices: List[int],
        physical_dims: int = 2,
        name: str = "MPS",
        dtype: np.dtype = np.complex128,
    ) -> "MPS":
        """
        Create a product MPS from computational basis labels.

        Example:
            state_indices=[0,1,0,1] -> |0101>
        """
        L = len(state_indices)
        bond_dims = [1] * (L + 1)
        mps = cls(
            L=L,
            physical_dims=physical_dims,
            bond_policy=bond_dims,
            name=name,
            dtype=dtype,
        )

        for i, s in enumerate(state_indices):
            s = int(s)
            if not (0 <= s < physical_dims):
                raise ValueError(f"Invalid local state index at site {i}: {s}")
            mps.tensors[i].data[...] = 0
            mps.tensors[i].data[0, s, 0] = 1.0

        return mps

    @classmethod
    def from_local_states(
        cls,
        local_states: List[np.ndarray],
        name: str = "MPS",
        dtype: np.dtype = np.complex128,
    ) -> "MPS":
        """
        Create a product MPS from arbitrary local statevectors.

        Args:
            local_states[i]: 1D array of shape (d_i,), not necessarily normalized.

        Returns:
            Product MPS with bond dims all 1.
        """
        if len(local_states) == 0:
            raise ValueError("local_states must be a non-empty list")

        physical_dims = [int(np.asarray(v).shape[0]) for v in local_states]
        L = len(physical_dims)
        bond_dims = [1] * (L + 1)

        mps = cls(
            L=L,
            physical_dims=physical_dims,
            bond_policy=bond_dims,
            name=name,
            dtype=dtype,
        )

        for i, v in enumerate(local_states):
            v = np.asarray(v, dtype=dtype).reshape(-1)
            if v.shape[0] != physical_dims[i]:
                raise ValueError(f"local_states[{i}] has wrong length")

            mps.tensors[i].data[...] = 0
            mps.tensors[i].data[0, :, 0] = v

        return mps

    # -------------------------
    # Internal helpers
    # -------------------------

    def _parse_physical_dims(self, physical_dims: PhysDims) -> List[int]:
        if isinstance(physical_dims, int):
            if physical_dims <= 0:
                raise ValueError("physical_dims must be positive")
            return [int(physical_dims)] * self.L

        if len(physical_dims) != self.L:
            raise AssertionError("physical_dims must have length L")
        dims = [int(d) for d in physical_dims]
        if any(d <= 0 for d in dims):
            raise ValueError("All physical dimensions must be positive")
        return dims

    def _resolve_bond_dims(self, bond_policy: BondPolicy, truncation: TruncationPolicy | None) -> List[int]:
        # Explicit bond dims
        if isinstance(bond_policy, list):
            if len(bond_policy) != self.L + 1:
                raise AssertionError("Explicit bond_policy list must have length L+1")
            dims = [int(x) for x in bond_policy]
            if any(d <= 0 for d in dims):
                raise ValueError("Bond dimensions must be positive")
            return dims

        # Helper: extract chi_max if present
        chi_max = None
        if truncation is not None and hasattr(truncation, "chi_max"):
            chi_max = getattr(truncation, "chi_max")

        if bond_policy == "uniform":
            if chi_max is None:
                raise ValueError("bond_policy='uniform' requires truncation.chi_max to be set")
            chi = int(chi_max)
            if chi <= 0:
                raise ValueError("truncation.chi_max must be positive for uniform bond policy")
            return [1] + [chi] * (self.L - 1) + [1]

        if bond_policy == "default":
            # chi_i = min(prod left physical dims, prod right physical dims)
            dims: List[int] = [1]
            left_prod = 1
            right_prod = 1
            for d in self._physical_dims:
                right_prod *= d

            for d in self._physical_dims:
                left_prod *= d
                right_prod //= d
                chi = min(left_prod, right_prod)
                if chi_max is not None:
                    chi = min(int(chi), int(chi_max))
                dims.append(int(chi))
            return dims

        raise ValueError(f"Unknown bond_policy: {bond_policy!r}")

    def _create_empty_tensors(self, dtype: np.dtype) -> None:
        """Create site tensors with indices [bond_left, physical, bond_right]."""
        self.tensors = []
        for i in range(self.L):
            inds = [self.bonds[i], self.indices[i], self.bonds[i + 1]]
            if allocate:
                shape = (inds[0].dim, inds[1].dim, inds[2].dim)
                data = np.zeros(shape, dtype=dtype)
            else:
                data = None
            self.tensors.append(Tensor(data, indices=inds))

    def _assert_materialized(self) -> None:
        for i, t in enumerate(self.tensors):
            if t.data is None:
                raise ValueError(f"MPS tensor at site {i} has data=None (structure-only MPS)")

    # -------------------------
    # Python protocol
    # -------------------------

    def __len__(self) -> int:
        return self.L

    def __repr__(self) -> str:
        return f"MPS(L={self.L}, phys_dims={self.physical_dims}, bond_dims={self.bond_dims})"

    # -------------------------
    # Basic properties
    # -------------------------

    @property
    def bond_dims(self) -> List[int]:
        """Bond dimensions between sites (including boundaries), length L+1."""
        return [b.dim for b in self.bonds]

    @property
    def physical_dims(self) -> List[int]:
        """Physical dimension at each site, length L."""
        return [p.dim for p in self.indices]

    # -------------------------
    # Core linear-algebra helpers
    # -------------------------

    def norm(self) -> float:
        """
        Compute the norm of the MPS as √⟨ψ|ψ⟩.

        Algorithm:
            Contract the MPS with its conjugate from left to right.
        """
        self._assert_materialized()

        overlap = np.eye(self.bond_dims[0], dtype=np.complex128)

        for tensor in self.tensors:
            # overlap: (chi_left, chi_left')
            # tensor: (chi_left, d, chi_right)

            temp = np.tensordot(overlap, tensor.data, axes=([0], [0]))
            # temp: (chi_left', d, chi_right)

            temp = np.tensordot(temp, tensor.conj().data, axes=([0, 1], [0, 1]))
            # temp: (chi_right, chi_right')

            overlap = temp

        return float(np.sqrt(np.abs(np.trace(overlap))))

    def normalize(self) -> "MPS":
        """
        Normalize the MPS in-place so that ⟨ψ|ψ⟩ = 1.

        Returns:
            self for chaining.
        """
        self._assert_materialized()

        nrm = self.norm()
        if nrm > 0:
            self.tensors[0] = self.tensors[0] * (1.0 / nrm)
        return self

    def copy(self) -> "MPS":
        """Creates a deep copy of the MPS."""
        return MPS.from_tensors(self.tensors, name=self.name)

    def to_dense(self) -> np.ndarray:
        """
        Convert the MPS to a full statevector |Ψ> as a 1D array of length Π_i d_i.

        Intended for small systems and for checks/debugging.
        """
        self._assert_materialized()

        psi = self.tensors[0].data  # (chiL, d0, chiR)

        for t in self.tensors[1:]:
            psi = np.tensordot(psi, t.data, axes=([psi.ndim - 1], [0]))
            # grows physical legs, keeps boundary bonds on ends

        psi = np.squeeze(psi)  # remove boundary dims if 1
        return psi.reshape(-1)
