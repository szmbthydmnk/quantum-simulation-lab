from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from .policy import TruncationPolicy

boundary_condition = Literal["open", "periodic"]
system_type = Literal["qubit", "spin-1/2"]   # qubit and spin-1/2 identical for now;
                                              # fermionic distinction + qudit-# planned


@dataclass
class Environment:
    """
    Centralized configuration for a 1D quantum lattice system.

    Attributes:
        L:              Number of sites.
        d:              Local Hilbert space dimension (2 for qubits/spin-1/2).
        system_type:    'qubit' or 'spin-1/2'.
        truncation_tol: SVD singular-value cutoff fallback.
        bc:             Boundary condition: 'open' or 'periodic'.
        max_bond_dim:   Hard cap on bond dimension.
        complex_dtype:  Default numpy dtype for tensor data.
        truncation:     Optional explicit TruncationPolicy. When set,
                        algorithms use this over raw truncation_tol/max_bond_dim.
        name:           Optional human-readable label.
    """

    L: int
    d: int

    system_type: str                       = "qubit"
    truncation_tol: float                  = 1e-10
    bc: boundary_condition                 = "open"
    max_bond_dim: int                      = 64
    complex_dtype: type                    = complex
    truncation: Optional[TruncationPolicy] = field(default=None, repr=False)
    name: str                              = ""

    def __post_init__(self) -> None:
        if self.L < 1:
            raise ValueError(f"L must be >= 1, got {self.L}")
        if self.d < 2:
            raise ValueError(f"d must be >= 2, got {self.d}")
        if self.bc not in ("open", "periodic"):
            raise ValueError(f"bc must be 'open' or 'periodic', got {self.bc!r}")
        if self.complex_dtype is complex:
            self.complex_dtype = np.complex128

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def qubit_chain(cls, L: int, chi_max: Optional[int] = None, **kwargs) -> "Environment":
        """Shorthand for a spin-1/2 / qubit (d=2) open-boundary chain."""
        mbd   = chi_max if chi_max is not None else 64
        trunc = TruncationPolicy(max_bond_dim=chi_max) if chi_max is not None else None
        return cls(L=L, d=2, bc="open", max_bond_dim=mbd, truncation=trunc, **kwargs)

    @classmethod
    def spin1_chain(cls, L: int, chi_max: Optional[int] = None, **kwargs) -> "Environment":
        """Shorthand for a spin-1 (d=3) open-boundary chain."""
        mbd   = chi_max if chi_max is not None else 64
        trunc = TruncationPolicy(max_bond_dim=chi_max) if chi_max is not None else None
        return cls(L=L, d=3, bc="open", max_bond_dim=mbd, truncation=trunc, **kwargs)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def hilbert_dim(self) -> int:
        """Full Hilbert space dimension d^L."""
        return self.d ** self.L

    @property
    def effective_truncation(self) -> TruncationPolicy:
        """
        Return the TruncationPolicy to use in algorithms.

        Prefers the explicit `truncation` field; falls back to constructing
        one from max_bond_dim + truncation_tol.
        """
        if self.truncation is not None:
            return self.truncation
        return TruncationPolicy(max_bond_dim=self.max_bond_dim, cutoff=self.truncation_tol)

    def __repr__(self) -> str:
        chi_str = f", chi_max={self.max_bond_dim}" if self.max_bond_dim else ""
        return (
            f"Environment(L={self.L}, d={self.d}, bc='{self.bc}'"
            f", system_type='{self.system_type}'{chi_str})"
        )
