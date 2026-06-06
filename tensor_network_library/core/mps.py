"""
Matrix Product State (MPS) implementation.

An MPS represents a many-body quantum state as a chain of rank-3 tensors::

    |ψ⟩ = Σ  A[0]^{s₀} A[1]^{s₁} … A[L-1]^{s_{L-1}} |s₀ s₁ … s_{L-1}⟩

Axis / index conventions (Schollwöck notation):
    site tensor shape : (χ_left, d, χ_right)
        axis 0 : left  virtual bond  (χ_left)
        axis 1 : physical index      (d — local Hilbert-space dimension)
        axis 2 : right virtual bond  (χ_right)
    Boundary sites have χ_left = 1 (leftmost) or χ_right = 1 (rightmost).

Key operations:
    - :meth:`MPS.product_state`  – factory for unentangled product states
    - :meth:`MPS.random`         – factory for random MPS (useful initialisation)
    - :meth:`MPS.to_dense`       – convert to full state vector (small L only)
    - :meth:`MPS.norm`           – compute ‖ψ‖ via full contraction
    - :meth:`MPS.copy`           – deep copy preserving all metadata
"""
from __future__ import annotations

from typing import List, Union, Sequence
import numpy as np
from numpy.typing import NDArray

from .tensor import Tensor
from .index import Index
from .policy import TruncationPolicy

ComplexArray = NDArray[np.complex128]


class MPS:
    """
    Matrix Product State (MPS) for 1D quantum lattice systems.

    Represents a quantum state as a product of rank-3 tensors (one per site)
    with open boundary conditions.

    Attributes:
        L (int): Number of sites.
        d (int): Physical dimension per site.
        tensors (List[Tensor]): Site tensors, each of shape (chi_l, d, chi_r).
        bond_dims (List[int]): Bond dimensions [chi_0, chi_1, ..., chi_L].
        dtype (np.dtype): Data type of tensors.
        name (str): Optional human-readable label.
    """

    def __init__(
        self,
        L: int,
        d: int,
        bond_dims: Union[int, List[int]] = 1,
        dtype: np.dtype = np.complex128,
        name: str = "",
    ):
        """
        Initialize an MPS.

        Args:
            L: Chain length.
            d: Physical dimension (e.g. 2 for qubits).
            bond_dims: Bond dimensions. int for uniform bond dimension,
                       or list of length L+1 for explicit specification.
            dtype: Data type of tensors.
            name: Optional label.
        """
        self.L = L
        self.d = d
        self.dtype = dtype
        self.name = name

        # Bond dims
        if isinstance(bond_dims, int):
            self.bond_dims = [1] + [bond_dims] * (L - 1) + [1]
        else:
            if len(bond_dims) != L + 1:
                raise ValueError(
                    f"bond_dims list must have length L+1={L + 1}, got {len(bond_dims)}"
                )
            self.bond_dims = list(bond_dims)

        # Initialize site tensors
        self.tensors: List[Tensor] = []
        for i in range(L):
            chi_l = self.bond_dims[i]
            chi_r = self.bond_dims[i + 1]
            data = np.zeros((chi_l, d, chi_r), dtype=dtype)

            left_ind = Index(dim=chi_l, name=f"L{i}", tags={f"site_{i}", "left"})
            phys_ind = Index(dim=d,     name=f"P{i}", tags={f"site_{i}", "physical"})
            right_ind = Index(dim=chi_r, name=f"R{i}", tags={f"site_{i}", "right"})

            self.tensors.append(Tensor(data, indices=[left_ind, phys_ind, right_ind]))

    # -------------------------
    # Factory methods
    # -------------------------

    @classmethod
    def product_state(
        cls,
        local_states: List[np.ndarray],
        dtype: np.dtype = np.complex128,
        name: str = "product_state",
    ) -> "MPS":
        """
        Create a product-state MPS from a list of local state vectors.

        Args:
            local_states: List of L arrays, each of shape (d,) or (d, 1).
                          Each array is the local quantum state at that site.
            dtype: Data type.
            name: Optional label.

        Returns:
            MPS representing the tensor product of all local states.
        """
        L = len(local_states)
        d = len(local_states[0].reshape(-1))

        mps = cls(L=L, d=d, bond_dims=1, dtype=dtype, name=name)

        for i, v in enumerate(local_states):
            v = np.asarray(v, dtype=dtype).reshape(-1)
            if len(v) != d:
                raise ValueError(
                    f"Local state at site {i} has dimension {len(v)}, expected {d}."
                )
            # Shape (1, d, 1)
            mps.tensors[i].data = v.reshape(1, d, 1)

        return mps

    @classmethod
    def random(
        cls,
        L: int,
        d: int,
        chi: int,
        dtype: np.dtype = np.complex128,
        seed: int | None = None,
        name: str = "random_mps",
    ) -> "MPS":
        """
        Create a random MPS.

        Args:
            L: Chain length.
            d: Physical dimension.
            chi: Bond dimension.
            dtype: Data type.
            seed: Random seed.
            name: Optional label.

        Returns:
            Random (unnormalized) MPS.
        """
        rng = np.random.default_rng(seed)
        bond_dims = [1] + [chi] * (L - 1) + [1]
        mps = cls(L=L, d=d, bond_dims=bond_dims, dtype=dtype, name=name)

        for i in range(L):
            chi_l = bond_dims[i]
            chi_r = bond_dims[i + 1]
            data = rng.standard_normal((chi_l, d, chi_r)).astype(dtype)
            if np.issubdtype(dtype, np.complexfloating):
                data = data + 1j * rng.standard_normal((chi_l, d, chi_r)).astype(dtype)
            mps.tensors[i].data = data

        return mps

    @classmethod
    def from_statevector(
        cls,
        psi: np.ndarray,
        physical_dims: Union[int, List[int]] = 2,
        truncation: TruncationPolicy | None = None,
        absorb: str = "right",
        normalize: bool = False,
        dtype: np.dtype = np.complex128,
        name: str = "statevector_mps",
    ) -> "MPS":
        """
        Convert a dense statevector to an exact MPS via sequential SVDs.

        Args:
            psi: Dense statevector of shape (d**L,) or (d, d, ..., d).
            physical_dims: Physical dimension per site (int) or list of d_i.
            truncation: Optional truncation policy.
            absorb: How to absorb singular values: 'right', 'left', or 'sqrt'.
            normalize: Normalize the state if True.
            dtype: Data type.
            name: Optional label.

        Returns:
            MPS representing the input statevector.
        """
        psi = np.asarray(psi, dtype=dtype)

        # Determine L and d
        if psi.ndim == 1:
            if isinstance(physical_dims, int):
                d = physical_dims
                L = int(np.round(np.log(len(psi)) / np.log(d)))
                if d**L != len(psi):
                    raise ValueError(
                        f"Statevector length {len(psi)} is not a power of d={d}."
                    )
                dims = [d] * L
            else:
                dims = list(physical_dims)
                L = len(dims)
                if np.prod(dims) != len(psi):
                    raise ValueError(
                        f"Product of physical_dims {dims} doesn't match statevector length {len(psi)}."
                    )
            psi = psi.reshape(dims)
        else:
            dims = list(psi.shape)
            L = len(dims)

        if normalize:
            norm = np.linalg.norm(psi)
            if norm > 0:
                psi = psi / norm

        bond_dims = [1] * (L + 1)
        tensors_data = []

        block = psi.reshape(1, -1)  # (1, d_0 * d_1 * ... * d_{L-1})

        for i in range(L - 1):
            d_i = dims[i]
            rows = block.shape[0]
            block = block.reshape(rows * d_i, -1)

            from scipy.linalg import svd as scipy_svd
            U, S, Vh = scipy_svd(block, full_matrices=False, lapack_driver="gesdd")

            chi = len(S)
            if truncation is not None:
                chi = min(truncation.choose_bond_dim(S), chi)
                U = U[:, :chi]
                S = S[:chi]
                Vh = Vh[:chi, :]

            bond_dims[i + 1] = chi

            # Absorb singular values
            if absorb == "right":
                A = U.reshape(bond_dims[i], d_i, chi)
                block = np.diag(S) @ Vh
            elif absorb == "left":
                A = (U * S[np.newaxis, :]).reshape(bond_dims[i], d_i, chi)
                block = Vh
            elif absorb == "sqrt":
                sqrt_S = np.sqrt(np.abs(S))
                A = (U * sqrt_S[np.newaxis, :]).reshape(bond_dims[i], d_i, chi)
                block = (np.diag(sqrt_S) @ Vh)
            else:
                raise ValueError(f"Unknown absorb mode: {absorb!r}")

            tensors_data.append(A)

        # Last site
        bond_dims[L] = 1
        d_last = dims[-1]
        A_last = block.reshape(bond_dims[L - 1], d_last, 1)
        tensors_data.append(A_last)

        mps = cls(L=L, d=dims[0], bond_dims=bond_dims, dtype=dtype, name=name)
        for i, A in enumerate(tensors_data):
            mps.tensors[i].data = A.astype(dtype)

        return mps

    @classmethod
    def from_tensors(
        cls,
        tensors: List[Tensor],
        name: str = "",
    ) -> "MPS":
        """
        Build an MPS from an explicit list of site tensors.

        Args:
            tensors: List of Tensor objects, each of shape (chi_l, d, chi_r).
            name: Optional label.

        Returns:
            MPS wrapping the provided tensors.
        """
        L = len(tensors)
        if L == 0:
            raise ValueError("Cannot build MPS from empty tensor list.")

        d = tensors[0].shape[1]
        bond_dims = [t.shape[0] for t in tensors] + [tensors[-1].shape[2]]
        dtype = tensors[0].data.dtype if tensors[0].data is not None else np.complex128

        mps = cls(L=L, d=d, bond_dims=bond_dims, dtype=dtype, name=name)
        mps.tensors = list(tensors)
        return mps

    # -------------------------
    # Python protocol
    # -------------------------

    def __len__(self) -> int:
        return self.L

    def __repr__(self) -> str:
        shapes_str = ", ".join(str(t.shape) for t in self.tensors)
        return f"MPS(L={self.L}, d={self.d}, name='{self.name}', shapes=[{shapes_str}])"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, idx: int) -> Tensor:
        return self.tensors[idx]

    # -------------------------
    # Core operations
    # -------------------------

    def copy(self) -> "MPS":
        """Create a deep copy of the MPS."""
        new_mps = MPS(L=self.L, d=self.d, bond_dims=self.bond_dims.copy(), dtype=self.dtype, name=self.name)
        new_mps.tensors = [t.copy() for t in self.tensors]
        return new_mps

    def to_dense(self) -> np.ndarray:
        """
        Convert MPS to a dense statevector.

        Contracts all tensors and returns the full state vector.
        Only feasible for small L (exponential memory).

        Returns:
            Dense statevector of shape (d**L,).
        """
        # Start from the leftmost tensor
        result = self.tensors[0].data  # (1, d, chi_r)
        result = result.reshape(self.d, -1)  # (d, chi_r)

        for i in range(1, self.L):
            A = self.tensors[i].data  # (chi_l, d, chi_r)
            chi_l, d_i, chi_r = A.shape
            # Einsum: result(s_{0..i-1}, chi_l) x A(chi_l, s_i, chi_r) -> result(s_{0..i}, chi_r)
            result = np.tensordot(result, A, axes=([-1], [0]))
            # result shape: (d, ..., d, chi_r)
            result = result.reshape(-1, chi_r)

        # result shape: (d**L, 1) -> (d**L,)
        return result.reshape(-1)

    def norm(self) -> float:
        """
        Compute the norm of the MPS state vector.

        Returns:
            float: The norm ||psi||.
        """
        psi = self.to_dense()
        return float(np.linalg.norm(psi))

    def normalize(self) -> "MPS":
        """
        Normalize the MPS state vector in-place.

        Returns:
            self (for chaining)
        """
        n = self.norm()
        if n > 0:
            self.tensors[0].data /= n
        return self

    def inner(self, other: "MPS") -> complex:
        """
        Compute inner product <self|other>.

        Uses sequential contraction (O(L chi^3 d) cost).

        Args:
            other: Ket MPS.

        Returns:
            Complex inner product <self|other>.
        """
        if self.L != other.L:
            raise ValueError("MPS lengths must match for inner product.")

        # Start from the left boundary: L_0 = [[1]] shape (1,1)
        L_env = np.ones((1, 1), dtype=self.dtype)

        for i in range(self.L):
            A = self.tensors[i].data    # (chi_l, d, chi_r)
            B = other.tensors[i].data   # (chi_l, d, chi_r)

            # Contract: L_env(a,b) A*(a,s,c) B(b,s,d) -> new_L(c,d)
            # Step 1: L_env(a,b) B(b,s,d) -> temp(a,s,d)
            temp = np.tensordot(L_env, B, axes=([1], [0]))
            # Step 2: A*(a,s,c) temp(a,s,d) -> result(c,d)
            L_env = np.tensordot(A.conj(), temp, axes=([0, 1], [0, 1]))

        # L_env is now (1,1): extract scalar
        return complex(L_env[0, 0])

    def overlap(self, other: "MPS") -> float:
        """
        Compute |<self|other>|^2.

        Args:
            other: Another MPS.

        Returns:
            float: |<self|other>|^2.
        """
        return float(abs(self.inner(other))**2)

    # -------------------------
    # Properties
    # -------------------------

    @property
    def shape(self) -> List[tuple]:
        """Returns shapes of all site tensors."""
        return [t.shape for t in self.tensors]

    @property
    def physical_dims(self) -> List[int]:
        """Physical dimensions of all sites."""
        return [t.shape[1] for t in self.tensors]

    @property
    def max_bond_dim(self) -> int:
        """Maximum bond dimension across all bonds."""
        return max(self.bond_dims)

    @property
    def dtype(self) -> np.dtype:
        """Data type of tensors."""
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = np.dtype(value)
