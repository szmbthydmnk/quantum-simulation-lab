from __future__ import annotations

import numpy as np

from .mps import MPS
from .mpo import MPO

def expectation_value(mps: MPS, mpo: MPO) -> float:
    """
    Compute <psi|H|psi> for an MPS and Hamiltonian MPO.

    This is intended for small systems (tests, debugging), since it
    goes via dense state vectors.

    Args:
        mps: State |psi>, as an MPS.
        mpo: Hamiltonian H, as an MPO.

    Returns:
        The real-valued expectation <psi|H|psi>.

    Disclaimer this is for test and development purposes, this goes as d^L: Not efficient.
    """
    # Sanity checks in terms of structure
    if len(mps) != mpo.L:
        raise ValueError(
            f"Length mismatch: MPS has L={len(mps)}, MPO has L={mpo.L}"
        )
    if mps.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"Physical dimension mismatch: MPS has {mps.physical_dims}, "
            f"MPO has {mpo.physical_dims}"
        )
    
    # H |psi>
    psi_H = mpo.apply(mps)

    # convert both to dense:
    v = mps.to_dense()
    Hv = psi_H.to_dense()

    return float(np.vdot(v, Hv))