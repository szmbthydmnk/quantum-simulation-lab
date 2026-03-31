from __future__ import annotations

import numpy as np

from .mps import MPS
from .mpo import MPO


def expectation_value(mps: MPS, mpo: MPO) -> float:
    """Compute <psi|H|psi> for an MPS and Hamiltonian MPO.

    This helper is intended for small systems (tests, debugging), since it
    goes via dense state vectors and therefore scales as O(d^L).

    Args:
        mps: State |psi>, as an MPS.
        mpo: Hamiltonian H, as an MPO.

    Returns:
        The real-valued expectation <psi|H|psi>.
    """
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

    # Convert both to dense
    v = mps.to_dense()
    Hv = psi_H.to_dense()

    return float(np.vdot(v, Hv))


# ---------------------------------------------------------------------------
# Environment-based expectation value (efficient in bond dimension)
# ---------------------------------------------------------------------------


def _update_left_env(L_i: np.ndarray, A_i: np.ndarray, W_i: np.ndarray) -> np.ndarray:
    """Update the left environment by absorbing site i.

    Shapes:
        L_i: (chiL, chiL, wL)
        A_i: (chiL, d,    chiR)
        W_i: (wL,  d, d,  wR)
    Returns:
        L_{i+1}: (chiR, chiR, wR)
    """
    return np.einsum(
        "abx,aic,bjd,xijy->cdy",
        L_i,
        A_i,
        A_i.conj(),
        W_i,
        optimize=True,
    )


def _update_right_env(R_ip1: np.ndarray, A_i: np.ndarray, W_i: np.ndarray) -> np.ndarray:
    """Update the right environment by absorbing site i from the right.

    Shapes:
        R_{i+1}: (chiR, chiR, wR)
        A_i:     (chiL, d,    chiR)
        W_i:     (wL,  d, d,  wR)
    Returns:
        R_i:     (chiL, chiL, wL)
    """
    return np.einsum(
        "cdy,aic,bjd,xijy->abx",
        R_ip1,
        A_i,
        A_i.conj(),
        W_i,
        optimize=True,
    )


def build_left_environments(mps: MPS, mpo: MPO) -> list[np.ndarray]:
    """Build all left environments L[i] for i=0..L.

    L[0] corresponds to the left boundary (no sites contracted yet) and has
    shape (1, 1, 1). L[i+1] is obtained from L[i] by absorbing site i.
    """
    if len(mps) != mpo.L:
        raise ValueError(
            f"Length mismatch: MPS has L={len(mps)}, MPO has L={mpo.L}"
        )
    if mps.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"Physical dimension mismatch: MPS has {mps.physical_dims}, "
            f"MPO has {mpo.physical_dims}"
        )

    L = len(mps)
    dtype = mps.dtype if hasattr(mps, "dtype") else mpo.dtype

    L_env: list[np.ndarray] = []
    # Left boundary: trivial environment
    L0 = np.ones((1, 1, 1), dtype=dtype)
    L_env.append(L0)

    for i in range(L):
        A_i = mps.tensors[i].data
        if A_i is None:
            raise ValueError(f"MPS tensor at site {i} has data=None")
        W_i = mpo.tensors[i].data
        L_next = _update_left_env(L_env[i], A_i, W_i)
        L_env.append(L_next)

    return L_env


def build_right_environments(mps: MPS, mpo: MPO) -> list[np.ndarray]:
    """Build all right environments R[i] for i=0..L.

    R[L] corresponds to the right boundary (no sites contracted yet) and has
    shape (1, 1, 1). R[i] is obtained from R[i+1] by absorbing site i.
    """
    if len(mps) != mpo.L:
        raise ValueError(
            f"Length mismatch: MPS has L={len(mps)}, MPO has L={mpo.L}"
        )
    if mps.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"Physical dimension mismatch: MPS has {mps.physical_dims}, "
            f"MPO has {mpo.physical_dims}"
        )

    L = len(mps)
    dtype = mps.dtype if hasattr(mps, "dtype") else mpo.dtype

    R_env: list[np.ndarray] = [None] * (L + 1)
    # Right boundary: trivial environment
    R_env[L] = np.ones((1, 1, 1), dtype=dtype)

    for i in reversed(range(L)):
        A_i = mps.tensors[i].data
        if A_i is None:
            raise ValueError(f"MPS tensor at site {i} has data=None")
        W_i = mpo.tensors[i].data
        R_env[i] = _update_right_env(R_env[i + 1], A_i, W_i)

    return R_env


def expectation_value_env(mps: MPS, mpo: MPO) -> float:
    """Efficient expectation value <psi|H|psi> using environments.

    This contracts the MPS and MPO using left and right environments and does
    not form dense statevectors or operators. The cost scales polynomially in
    bond dimensions and system size, as in standard MPS/MPO algorithms.
    """
    if len(mps) == 0:
        raise ValueError("MPS must have at least one site")

    L_env = build_left_environments(mps, mpo)
    R_env = build_right_environments(mps, mpo)

    # Use site 0; for a fully built environment this gives the full energy.
    A0 = mps.tensors[0].data
    if A0 is None:
        raise ValueError("MPS tensor at site 0 has data=None")
    W0 = mpo.tensors[0].data

    L0 = L_env[0]      # (1,1,1) at left boundary
    R1 = R_env[1]      # environment to the right of site 0

    # E = Σ L0(a,b,x) A0(a,i,c) A0*(b,j,d) W0(x,i,j,y) R1(c,d,y)
    E = np.einsum(
        "abx,aic,bjd,xijy,cdy->",
        L0,
        A0,
        A0.conj(),
        W0,
        R1,
        optimize=True,
        )
    return float(np.real_if_close(E))
