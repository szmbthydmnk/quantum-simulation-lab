"""Matrix Product State (MPS) implementation."""

from __future__ import annotations

from typing import List, Union, Sequence
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
        - By default the MPS is created as a *structure only* object:
            tensors have data=None but valid Index connectivity.
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
        Initialize an MPS *structure* (Indices + site tensors with data=None).

        Args:
            L: Chain length.
            physical_dims: Physical dimension(s). If int, same for all sites.
                If List, must have length L.
            bond_policy:
                - "default": chi_i = min(prod_{k<=i} d_k, prod_{k>i} d_k),
                            optionally capped by truncation.max_bond_dim.
                - "uniform": uses truncation.max_bond_dim as internal chi (boundaries stay 1).
                - List[int]: explicit bond dims of length L+1 (boundaries should be 1).
            name: Name tag for this MPS (used in Index names).
            truncation: Optional truncation policy (used to cap bond dims for "default"/"uniform").
            dtype: Default dtype used when materializing tensors.
        """
        if L <= 0:
            raise ValueError("L must be a positive integer")

        self.L = int(L)
        self.name = str(name)
        self.dtype = dtype

        # Indices (Index objects)
        self.indices: List[Index] = []  # physical
        self.bonds: List[Index] = []    # bonds

        # Site tensors (Tensor objects)
        self.tensors: List[Tensor] = []

        # Normalize physical dims to list[int]
        self._physical_dims: List[int] = self._parse_physical_dims(physical_dims)

        # Resolve bond dims (list[int] of length L+1)
        self._bond_dims: List[int] = self._resolve_bond_dims(
            bond_policy=bond_policy, truncation=truncation
        )

        # Create Index objects
        self.indices = [
            Index(
                dim=self._physical_dims[i],
                name=f"{self.name}_phys_{i}",
                tags=frozenset({"phys", f"i={i}"}),
            )
            for i in range(self.L)
        ]
        self.bonds = [
            Index(
                dim=self._bond_dims[i],
                name=f"{self.name}_bond_{i}",
                tags=frozenset({"bond", f"b={i}"}),
            )
            for i in range(self.L + 1)
        ]

        # Create unmaterialized site tensors
        self._create_empty_tensors()


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

        obj.indices = [t.indices[1] for t in obj.tensors]
        obj.bonds = [obj.tensors[0].indices[0]] + [t.indices[2] for t in obj.tensors]

        obj._physical_dims = [ix.dim for ix in obj.indices]
        obj._bond_dims = [ix.dim for ix in obj.bonds]

        # Best-effort dtype inference
        obj.dtype = np.complex128
        for t in obj.tensors:
            if t.data is not None:
                obj.dtype = t.data.dtype
                break

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

            mps.tensors[i].materialize_zeros(dtype=dtype)
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

            mps.tensors[i].materialize_zeros(dtype=dtype)
            mps.tensors[i].data[...] = 0
            mps.tensors[i].data[0, :, 0] = v

        return mps


    @classmethod
    def from_qubit_labels(cls,
                          labels: Sequence[str],
                          *,
                          name: str = "MPS",
                          dtype: np.dtype = np.complex128
                          ) -> "MPS":
        """
        Build a product-state MPS from 1-qubit labels like:
        ["0", "+", "i", "t3", "h7", "phi=pi/4"].
        """

        # Local import to avoid making core depend on "states" at import time
        from tensor_network_library.states.qubit_states import qubit_states

        local_vecs = [np.asarray(v, dtype=dtype) for v in qubit_states(labels)]
        return cls.from_local_states(local_states=local_vecs, name=name, dtype=dtype)


    @classmethod
    def from_random(
        cls,
        L: int,
        chi_max: int,
        physical_dims: PhysDims = 2,
        *,
        seed: int | None = None,
        name: str = "MPS",
        dtype: np.dtype = np.complex128,
    ) -> "MPS":
        """
        Create a random MPS with bond dimension ``chi_max`` and normalize it.

        The bond dims are capped by the maximum entanglement-entropy bond dim
        at each cut (i.e. ``min(d^i, d^(L-i), chi_max)``) so that boundary
        sites always have bond dim 1.  This matches the "default" bond policy
        with an explicit chi cap.

        The tensors are filled with iid complex Gaussian entries, then the
        whole MPS is normalized via :meth:`normalize`.

        Args:
            L:             Chain length.
            chi_max:       Maximum bond dimension (interior bonds).
            physical_dims: Local Hilbert space dimension(s).  Int for uniform.
            seed:          RNG seed for reproducibility.  ``None`` = random.
            name:          Name tag.
            dtype:         Complex dtype (default ``np.complex128``).

        Returns:
            Normalized random MPS.
        """
        rng = np.random.default_rng(seed)
        truncation = TruncationPolicy(max_bond_dim=chi_max)

        mps = cls(
            L=L,
            physical_dims=physical_dims,
            bond_policy="default",
            name=name,
            truncation=truncation,
            dtype=dtype,
        )

        for i in range(L):
            chi_l = mps._bond_dims[i]
            d_i   = mps._physical_dims[i]
            chi_r = mps._bond_dims[i + 1]
            data  = rng.standard_normal((chi_l, d_i, chi_r)).astype(dtype)
            if np.issubdtype(dtype, np.complexfloating):
                data = data + 1j * rng.standard_normal((chi_l, d_i, chi_r)).astype(dtype)
            mps.tensors[i] = Tensor(
                data,
                indices=[mps.bonds[i], mps.indices[i], mps.bonds[i + 1]],
            )

        mps.normalize()
        return mps


    @classmethod
    def from_statevector(cls,
                          psi: np.ndarray,
                          physical_dims: PhysDims = 2,
                          *,
                          name: str = "MPS",
                          truncation: TruncationPolicy | None = None,
                          absorb: str = "right",        #"right", "left", "sqrt"
                          normalize: bool = True,
                          dtype: np.dtype = np.complex128,
                          ) -> "MPS":
        """
        Build an MPS from a dense statevector via succesive SVD.

        If truncation is None: keep full rank at each cut (exact MPS)
        If truncation is set: keep truncation.choose_bond_dim(S) at each cut.
        """        

        vec = np.asarray(psi, dtype = dtype).reshape(-1)
        if vec.ndim != 1:
            raise ValueError("psi must be a 1-D array.")

        # Normalise physical dims
        if isinstance(physical_dims, int):
            d = physical_dims
            L = int(round(np.log(vec.size) / np.log(d)))
            if d ** L != vec.size:
                raise ValueError(
                    f"Statevector length {vec.size} is not a power of d={d}."
                )
            dims: List[int] = [d] * L
        else:
            dims = [int(x) for x in physical_dims]
            L = len(dims)
            expected = 1
            for x in dims: expected *= x
            if expected != vec.size:
                raise ValueError(
                    f"Product of physical_dims {dims} = {expected} != {vec.size}."
                )

        # Build structure (bond dims will be overwritten)
        mps = cls(L=L, physical_dims=dims, bond_policy="default",
                  truncation=truncation, name=name, dtype=dtype)

        # Successive SVD sweep left -> right
        M = vec.copy().reshape([1] + dims)   # shape: (1, d0, d1, ..., d_{L-1})
        chi_left = 1

        tensors_data: list[np.ndarray] = []

        for i in range(L - 1):
            d_i = dims[i]
            M   = M.reshape(chi_left * d_i, -1)     # merge left bond + phys

            U, S, Vh = np.linalg.svd(M, full_matrices=False)

            # Truncate
            if truncation is not None:
                k = truncation.choose_bond_dim(S)
                U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
            else:
                k = len(S)

            chi_right = k

            # Absorb singular values
            if absorb == "right":
                tensors_data.append(U.reshape(chi_left, d_i, chi_right))
                M = (np.diag(S) @ Vh)
            elif absorb == "left":
                tensors_data.append((U * S[np.newaxis, :]).reshape(chi_left, d_i, chi_right))
                M = Vh
            elif absorb == "sqrt":
                sqS = np.sqrt(S)
                tensors_data.append((U * sqS[np.newaxis, :]).reshape(chi_left, d_i, chi_right))
                M = (np.diag(sqS) @ Vh)
            else:
                raise ValueError(f"Unknown absorb mode: {absorb!r}")

            chi_left = chi_right

        # Last site
        tensors_data.append(M.reshape(chi_left, dims[-1], 1))

        # Build Index objects consistent with chosen bond dims
        bond_dims = [td.shape[0] for td in tensors_data] + [1]
        phys_indices = [
            Index(dim=dims[i], name=f"{name}_phys_{i}", tags=frozenset({"phys", f"i={i}"}))
            for i in range(L)
        ]
        bond_indices = [
            Index(dim=bond_dims[i], name=f"{name}_bond_{i}", tags=frozenset({"bond", f"b={i}"}))
            for i in range(L + 1)
        ]

        site_tensors = [
            Tensor(
                tensors_data[i],
                indices=[bond_indices[i], phys_indices[i], bond_indices[i + 1]],
            )
            for i in range(L)
        ]

        result = cls.from_tensors(site_tensors, name=name)

        if normalize:
            result.normalize()

        return result


    # -------------------------
    # Properties
    # -------------------------

    @property
    def physical_dims(self) -> List[int]:
        """List of physical dimensions."""
        return list(self._physical_dims)

    @property
    def bond_dims(self) -> List[int]:
        """List of bond dimensions (length L+1, boundaries typically 1)."""
        return list(self._bond_dims)


    # -------------------------
    # Materialization
    # -------------------------

    def _assert_materialized(self, site: int | None = None) -> None:
        """
        Raise RuntimeError if any (or a specific) site tensor has data=None.
        """
        if site is not None:
            if self.tensors[site].data is None:
                raise RuntimeError(f"Site {site} tensor is not materialized.")
        else:
            for i, t in enumerate(self.tensors):
                if t.data is None:
                    raise RuntimeError(f"Site {i} tensor is not materialized.")


    def materialize_zeros(self, dtype: np.dtype | None = None) -> None:
        """Fill all tensors with zeros (in-place)."""
        dt = dtype or self.dtype
        for t in self.tensors:
            t.materialize_zeros(dtype=dt)


    def materialize_random(self,
                           seed: int | None = None,
                           dtype: np.dtype | None = None) -> None:
        """Fill all tensors with random data, then normalize."""
        rng = np.random.default_rng(seed)
        dt  = dtype or self.dtype
        for t in self.tensors:
            shape = tuple(ix.dim for ix in t.indices)
            data  = rng.standard_normal(shape).astype(dt)
            if np.issubdtype(dt, np.complexfloating):
                data = data + 1j * rng.standard_normal(shape).astype(dt)
            t.data = data
        self.normalize()


    # -------------------------
    # Product states
    # -------------------------

    def product_state(self, state_indices: List[int]) -> None:
        """
        In-place: set this MPS to a product state |s0 s1 ... s_{L-1}>.

        Requires bond dims == 1 everywhere (chi=1 product state MPS).
        Materializes tensors with zeros then sets the correct entry to 1.
        """
        if any(bd != 1 for bd in self._bond_dims):
            raise ValueError("product_state() requires all bond dims == 1.")
        if len(state_indices) != self.L:
            raise ValueError(f"state_indices has length {len(state_indices)}, expected {self.L}.")

        for i, s in enumerate(state_indices):
            s = int(s)
            if not (0 <= s < self._physical_dims[i]):
                raise ValueError(f"Invalid state index {s} at site {i}")
            self.tensors[i].materialize_zeros()
            self.tensors[i].data[0, s, 0] = 1.0


    # -------------------------
    # Norm and normalization
    # -------------------------

    def norm(self) -> float:
        """
        Compute <psi|psi> via sequential contraction (left to right).

        Complexity: O(L * chi^2 * d).
        """
        self._assert_materialized()
        transfer = np.ones((1, 1), dtype=self.dtype)

        for t in self.tensors:
            A = t.data   # shape (chi_l, d, chi_r)
            # transfer_{a,b} = sum_{s,a',b'} transfer_{a',b'} * A_{a',s,chi_r_old} * conj(A)_{b',s,...}
            # Standard left contraction:
            # transfer' = einsum('ab, ais, bis -> ij', transfer, A, A.conj())
            tmp = np.tensordot(transfer, A, axes=([0], [0]))       # (b, d, chi_r)
            transfer = np.tensordot(tmp, A.conj(), axes=([0, 1], [0, 1]))  # (chi_r, chi_r)

        return float(np.sqrt(np.abs(transfer[0, 0])))


    def normalize(self) -> float:
        """
        Normalize the MPS in-place. Returns the previous norm.
        """
        self._assert_materialized()
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize a zero-norm MPS.")
        # Distribute 1/n into the last site tensor
        self.tensors[-1].data = self.tensors[-1].data / n
        return n


    # -------------------------
    # Measurement helpers
    # -------------------------

    def overlap(self, other: "MPS") -> complex:
        """
        Compute <self|other> via sequential contraction.
        """
        self._assert_materialized()
        other._assert_materialized()

        transfer = np.ones((1, 1), dtype=self.dtype)

        for A, B in zip(self.tensors, other.tensors):
            a = A.data
            b = B.data
            tmp = np.tensordot(transfer, a.conj(), axes=([0], [0]))  # (chi_r_A, d, chi_r_B)
            transfer = np.tensordot(tmp, b, axes=([0, 1], [0, 1]))   # wrong — fix:
            # Correct contraction:
            # Let's redo properly
            transfer = np.einsum('ab,ais,bis->ij', transfer, a.conj(), b)

        return complex(transfer[0, 0])


    # -------------------------
    # Utility
    # -------------------------

    def copy(self) -> "MPS":
        """Deep copy."""
        new_tensors = [t.copy() for t in self.tensors]
        return MPS.from_tensors(new_tensors, name=self.name)


    def __len__(self) -> int:
        return self.L


    def __repr__(self) -> str:
        bd = self._bond_dims
        pd = self._physical_dims
        materialized = all(t.data is not None for t in self.tensors)
        return (
            f"MPS(name={self.name!r}, L={self.L}, "
            f"physical_dims={pd}, max_bond={max(bd)}, "
            f"materialized={materialized})"
        )


    # -------------------------
    # Private helpers
    # -------------------------

    def _parse_physical_dims(self, physical_dims: PhysDims) -> List[int]:
        if isinstance(physical_dims, int):
            assert physical_dims >= 2, "Physical dimension must be >= 2"
            return [physical_dims] * self.L
        else:
            dims = list(physical_dims)
            assert len(dims) == self.L, (
                f"physical_dims list must have length L={self.L}, got {len(dims)}"
            )
            return [int(d) for d in dims]


    def _resolve_bond_dims(
        self,
        bond_policy: BondPolicy,
        truncation: TruncationPolicy | None,
    ) -> List[int]:
        """Compute the list of L+1 bond dims from the policy."""

        chi_max = truncation.max_bond_dim if truncation is not None else None

        if isinstance(bond_policy, list):
            bd = [int(x) for x in bond_policy]
            assert len(bd) == self.L + 1, (
                f"Explicit bond_policy list must have length L+1={self.L + 1}."
            )
            return bd

        if bond_policy == "uniform":
            if chi_max is None:
                raise ValueError('bond_policy="uniform" requires a TruncationPolicy with max_bond_dim.')
            return [1] + [chi_max] * (self.L - 1) + [1]

        # "default": chi_i = min(prod_{k<i} d_k, prod_{k>=i} d_k)
        dims = self._physical_dims
        left_prod  = [1] * (self.L + 1)
        right_prod = [1] * (self.L + 1)

        for i in range(1, self.L + 1):
            left_prod[i] = left_prod[i - 1] * dims[i - 1]
        for i in range(self.L - 1, -1, -1):
            right_prod[i] = right_prod[i + 1] * dims[i]

        bd = [min(left_prod[i], right_prod[i]) for i in range(self.L + 1)]

        if chi_max is not None:
            bd = [min(x, chi_max) for x in bd]

        return bd


    def _create_empty_tensors(self) -> None:
        """Create site tensors with data=None."""
        self.tensors = [
            Tensor(
                data=None,
                indices=[self.bonds[i], self.indices[i], self.bonds[i + 1]],
            )
            for i in range(self.L)
        ]
