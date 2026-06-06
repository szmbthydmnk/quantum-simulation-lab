"""
Truncation policy for SVD-based bond-dimension control.

Provides :class:`TruncationPolicy`, a lightweight dataclass that encapsulates
the two standard truncation criteria used throughout tensor-network algorithms:

* **Singular-value cutoff** – discard all singular values whose square is
  below a threshold (equivalent to a Schmidt-rank cutoff on the reduced
  density matrix).
* **Maximum bond dimension** – hard cap on the number of singular values kept
  regardless of their magnitude.

Example::

    policy = TruncationPolicy(max_bond_dim=64, cutoff=1e-10)
    chi = policy.choose_bond_dim(singular_values)
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class TruncationPolicy:
    """
    Decide how many singular values to keep.

    Attributes:
        max_bond_dim: Upper bound on kept singular values. None = no cap.
        cutoff:       Discard singular values whose square is smaller than
                      this threshold. Default 0 = keep all above machine zero.
        strict:       If True, raise when tolerance requires more than
                      max_bond_dim. If False (default), silently cap.
    """
    max_bond_dim: Optional[int] = None
    cutoff: float               = 0.0
    strict: bool                = False

    def choose_bond_dim(self, singular_values: np.ndarray) -> int:
        s = np.asarray(singular_values, dtype=float)
        keep_tol = int((s**2 >= self.cutoff).sum()) if self.cutoff > 0 else len(s)

        if self.max_bond_dim is not None and keep_tol > self.max_bond_dim:
            if self.strict:
                raise ValueError(
                    f"TruncationPolicy strict violation: needed {keep_tol} singular "
                    f"values to satisfy cutoff={self.cutoff}, "
                    f"but max_bond_dim={self.max_bond_dim}."
                )
            return self.max_bond_dim

        return keep_tol
