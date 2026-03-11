"""
SystemConfig: centralized configuration object for 1D quantum lattice systems.

Passed optionally into MPS/MPO constructors and algorithm drivers to avoid
scattering system parameters (L, d, boundary conditions, truncation) across
call sites.

Usage::

    cfg = SystemConfig(L=10, d=2, boundary="open", truncation=TruncationPolicy(chi_max=32))
    mps = MPS(L=cfg.L, physical_dims=cfg.d, truncation=cfg.truncation)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from .policy import TruncationPolicy

BoundaryCondition = Literal["open", "periodic"]


@dataclass
class SystemConfig:
    """
    Configuration for a 1D quantum lattice system.

    Attributes:
        L:          Chain length (number of sites). Must be >= 1.
        d:          Local Hilbert space dimension (e.g. 2 for qubits). Must be >= 2.
        boundary:   Boundary condition: 'open' (OBC) or 'periodic' (PBC).
                    Note: most algorithms in this library currently support OBC only.
        truncation: Optional TruncationPolicy applied by default when constructing
                    MPS with this config and in algorithm sweeps.
        dtype:      Default numpy dtype for tensor data.
        name:       Optional human-readable label for this system.
    """

    L: int
    d: int = 2
    boundary: BoundaryCondition = "open"
    truncation: Optional[TruncationPolicy] = field(default=None, repr=False)
    dtype: object = field(default=None, repr=False)   # np.dtype filled in __post_init__
    name: str = ""

    def __post_init__(self) -> None:
        import numpy as np
        if self.L < 1:
            raise ValueError(f"L must be >= 1, got {self.L}")
        if self.d < 2:
            raise ValueError(f"d must be >= 2, got {self.d}")
        if self.boundary not in ("open", "periodic"):
            raise ValueError(f"boundary must be 'open' or 'periodic', got {self.boundary!r}")
        if self.dtype is None:
            self.dtype = np.complex128

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def qubit_chain(cls, L: int, chi_max: Optional[int] = None, **kwargs) -> "SystemConfig":
        """Shorthand for a spin-1/2 (d=2) open-boundary chain."""
        trunc = TruncationPolicy(chi_max=chi_max) if chi_max is not None else None
        return cls(L=L, d=2, boundary="open", truncation=trunc, **kwargs)

    @classmethod
    def spin1_chain(cls, L: int, chi_max: Optional[int] = None, **kwargs) -> "SystemConfig":
        """Shorthand for a spin-1 (d=3) open-boundary chain."""
        trunc = TruncationPolicy(chi_max=chi_max) if chi_max is not None else None
        return cls(L=L, d=3, boundary="open", truncation=trunc, **kwargs)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def hilbert_dim(self) -> int:
        """Full Hilbert space dimension d^L."""
        return self.d ** self.L

    @property
    def max_bond_dim(self) -> Optional[int]:
        """Shortcut to truncation.chi_max if set."""
        return self.truncation.max_bond_dim if self.truncation is not None else None

    def __repr__(self) -> str:
        trunc_str = f", chi_max={self.max_bond_dim}" if self.max_bond_dim else ""
        return f"SystemConfig(L={self.L}, d={self.d}, boundary='{self.boundary}'{trunc_str})"