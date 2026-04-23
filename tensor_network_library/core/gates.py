from __future__ import annotations

from typing import Literal

import numpy as np

from .mps import MPS
from .tensor import Tensor
from .index import Index
from .policy import TruncationPolicy


def _assert_materialized(mps: MPS) -> None:
    """
    Local copy of the materialization check so I don't rely on MPS internals.
    """
    for i, t in enumerate(mps.tensors):
        if t.data is None:
            raise ValueError(f"MPS tensor at site {i} has data=None (unmaterialized MPS)")


def _choose_chi(S: np.ndarray, 
                truncation: TruncationPolicy | None) -> int:
    """
    Decide the kept bond dimension from singular values S.
    """
    chi_full = int(S.shape[0])
    if truncation is None:
        return chi_full

    chi = int(truncation.choose_bond_dim(S))
    if chi <= 0:
        raise ValueError(
            "TruncationPolicy chose chi=0 "
            "(cutoff too large or state near-zero on this cut)."
        )
    return min(chi, chi_full)


def apply_two_site_gate(mps: MPS,
                        U: np.ndarray,
                        i: int,
                        *,
                        truncation: TruncationPolicy | None = None,
                        absorb: Literal["right", "left", "sqrt"] = "right",
                        inplace: bool = False,
                        ) -> MPS:
    """
    Apply a nearest-neighbour two-site _unitary_ gate to an MPS bond.

    Gate acts on sites (i, i+1) with 0-based indexing (i = left site).

    Parameters
    ----------
    mps
        Input MPS. Must be fully materialized.
    U
        Two-site gate. Supported shapes:
          - (d, d, d, d) interpreted as U[s0', s1', s0, s1]
          - (d*d, d*d)   reshaped to the above.
        For now we assume d_i = d_{i+1} = d.
    i
        Left site index, 0 <= i < L-1.
    truncation
        Optional TruncationPolicy. If None, keep full SVD rank.
    absorb
        Where to absorb singular values:
          - "right": A carries U, B carries S Vh  (TEBD-style, S on right)
          - "left":  A carries U S, B carries Vh
          - "sqrt":  split sqrt(S) half/half across A and B
    inplace
        If True, modify the input MPS in-place and return it.
        If False (default), work on a copy and leave the input untouched.

    Returns
    -------
    MPS
        Updated MPS with a (possibly) new bond dimension at (i, i+1).

    Notes
    -----
    - This helper does *not* enforce a particular canonical gauge; it simply
      performs the standard TEBD update pattern: contract, apply gate, SVD,
      truncate, split.
    """
    
    if mps.L < 2:
        raise ValueError("Cannot apply a two-site gate to an MPS with L < 2.")

    if not (0 <= i < mps.L - 1):
        raise ValueError(f"Site index i={i} must satisfy 0 <= i < L-1 (L={mps.L}).")

    _assert_materialized(mps)

    # Work on a copy unless explicitly told otherwise
    if inplace:
        out = mps
    else:
        out = mps.copy()

    A = out.tensors[i].data  # shape (chi_l, d_i, chi_mid)
    B = out.tensors[i + 1].data  # shape (chi_mid, d_j, chi_r)

    if A is None or B is None:
        raise ValueError("Unmaterialized tensors encountered in apply_two_site_gate.")

    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("MPS site tensors must have shape (chi_l, d, chi_r).")

    chi_l, d_i, chi_mid = A.shape
    chi_mid_B, d_j, chi_r = B.shape

    if chi_mid_B != chi_mid:
        raise ValueError(
            f"Inconsistent bond dimension at sites {i} and {i+1}: "
            f"{chi_mid} vs {chi_mid_B}."
        )

    if d_i != d_j:
        raise NotImplementedError(
            "apply_two_site_gate currently assumes equal physical dims at i and i+1."
        )

    d = d_i

    # Normalize U to 4-index form U[s0', s1', s0, s1]
    U = np.asarray(U, dtype=out.dtype)
    if U.shape == (d * d, d * d):
        U = U.reshape(d, d, d, d)
    elif U.shape == (d, d, d, d):
        pass
    else:
        raise ValueError(
            f"Unsupported gate shape {U.shape!r}; "
            "expected (d*d, d*d) or (d, d, d, d)."
        )

    # Build theta[chi_l, s0, s1, chi_r] = contraction of A and B over intermediate bond
    theta = np.tensordot(A, B, axes=([2], [0]))  # (chi_l, d, chi_mid) x (chi_mid, d, chi_r)
    # -> (chi_l, d, d, chi_r)

    # Apply U on the two physical legs: result[o0, o1, chi_l, chi_r]
    theta = np.tensordot(U, theta, axes=([2, 3], [1, 2]))
    # Rearrange back to (chi_l, d, d, chi_r)
    theta = np.moveaxis(theta, (2, 3), (0, 3))

    # Reshape for SVD: group (chi_l, d) and (d, chi_r)
    theta_mat = theta.reshape(chi_l * d, d * chi_r)

    # SVD
    Umat, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)

    chi_new = _choose_chi(S, truncation)
    Umat = Umat[:, :chi_new]
    S = S[:chi_new]
    Vh = Vh[:chi_new, :]

    # Absorb singular values according to policy
    absorb = str(absorb).lower()
    if absorb not in {"right", "left", "sqrt"}:
        raise ValueError("absorb must be one of {'right', 'left', 'sqrt'}")

    if absorb == "right":
        # A' = Umat, B' = S Vh
        A_new = Umat.reshape(chi_l, d, chi_new)
        B_mat = S[:, None] * Vh  # diag(S) @ Vh
        B_new = B_mat.reshape(chi_new, d, chi_r)
    elif absorb == "left":
        # A' = Umat diag(S), B' = Vh
        A_mat = Umat * S[None, :]
        A_new = A_mat.reshape(chi_l, d, chi_new)
        B_new = Vh.reshape(chi_new, d, chi_r)
    else:  # "sqrt"
        s = np.sqrt(S)
        A_mat = Umat * s[None, :]
        B_mat = s[:, None] * Vh
        A_new = A_mat.reshape(chi_l, d, chi_new)
        B_new = B_mat.reshape(chi_new, d, chi_r)

    # Create / update the bond Index between i and i+1
    bond_new = Index(
        dim=chi_new,
        name=f"{out.name}_bond_{i+1}",
        tags=frozenset({"bond", f"b={i+1}"}),
    )
    out.bonds[i + 1] = bond_new

    # Replace site tensors with updated data and indices
    out.tensors[i] = Tensor(
        A_new.astype(out.dtype, copy=False),
        indices=[out.bonds[i], out.indices[i], bond_new],
    )
    out.tensors[i + 1] = Tensor(
        B_new.astype(out.dtype, copy=False),
        indices=[bond_new, out.indices[i + 1], out.bonds[i + 2]],
    )

    # Keep internal dimension bookkeeping consistent (used e.g. by bond_dims property)
    out._bond_dims = [ix.dim for ix in out.bonds]         # type: ignore[attr-defined]
    out._physical_dims = [ix.dim for ix in out.indices]   # type: ignore[attr-defined]

    return out