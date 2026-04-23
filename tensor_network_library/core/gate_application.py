# tensor_network_library/core/gate_application.py
"""Two-site gate application on MPS.

Implements the standard SVD-based two-site gate update used in TEBD:

    ┌───┐ ┌───┐           ┌──────┐
    │ A │─│ B │  ──U──>   │ A' │─│ B' │
    └───┘ └───┘           └──────┘

The two-site tensor is contracted with the gate U, then split back into
two tensors via SVD with optional truncation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

from .mps import MPS
from .tensor import Tensor
from .index import Index


# ---------------------------------------------------------------------------
# Core SVD split
# ---------------------------------------------------------------------------


def _svd_split(
    theta: np.ndarray,
    chiL: int,
    d: int,
    dR: int,
    chiR: int,
    max_bond: Optional[int],
    svd_cutoff: float,
    absorb: Literal["left", "right", "both"],
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-split a two-site tensor theta into left and right MPS tensors.

    Args:
        theta:      Two-site tensor of shape (chiL, d, dR, chiR).
        chiL:       Left bond dimension.
        d:          Physical dimension of left site.
        dR:         Physical dimension of right site.
        chiR:       Right bond dimension.
        max_bond:   Maximum bond dimension chi to keep (None = no limit).
        svd_cutoff: Discard singular values below this threshold.
        absorb:     Where to absorb singular values:
                      "left"  -> A' = U S,  B' = V†
                      "right" -> A' = U,    B' = S V†
                      "both"  -> A' = U √S, B' = √S V†
        normalize:  Renormalize singular values to unit norm after truncation.

    Returns:
        A_new: shape (chiL, d, chi_new)
        S:     shape (chi_new,)
        B_new: shape (chi_new, dR, chiR)
    """
    mat = theta.reshape(chiL * d, dR * chiR)

    U, s, Vh = np.linalg.svd(mat, full_matrices=False)

    keep = s > svd_cutoff
    n_keep = int(np.sum(keep))
    if max_bond is not None:
        n_keep = min(n_keep, max_bond)
    if n_keep == 0:
        n_keep = 1

    U = U[:, :n_keep]
    s = s[:n_keep]
    Vh = Vh[:n_keep, :]

    if normalize:
        norm = np.linalg.norm(s)
        if norm > 0.0:
            s = s / norm

    if absorb == "left":
        U = U * s[np.newaxis, :]
    elif absorb == "right":
        Vh = s[:, np.newaxis] * Vh
    else:  # "both"
        sqrt_s = np.sqrt(s)
        U = U * sqrt_s[np.newaxis, :]
        Vh = sqrt_s[:, np.newaxis] * Vh

    A_new = U.reshape(chiL, d, n_keep)
    B_new = Vh.reshape(n_keep, dR, chiR)

    return A_new, s, B_new


# ---------------------------------------------------------------------------
# Two-site gate application
# ---------------------------------------------------------------------------


def apply_two_site_gate(
    mps: MPS,
    gate: np.ndarray,
    site_i: int,
    *,
    max_bond: Optional[int] = None,
    svd_cutoff: float = 1e-12,
    absorb: Literal["left", "right", "both"] = "right",
    normalize: bool = False,
    inplace: bool = False,
) -> Tuple[MPS, np.ndarray]:
    """Apply a two-site unitary gate U to sites (site_i, site_i+1) of an MPS.

    The gate U acts on the physical indices of two adjacent sites. It must be
    provided as a matrix of shape (d*d, d*d) or as a rank-4 tensor of shape
    (d, d, d, d) with index ordering (i', j', i, j) where (i', j') are output
    (ket) indices and (i, j) are input (bra) indices.

    Args:
        mps:        Input MPS.
        gate:       Gate tensor; shape (d*d, d*d) or (d, d, d, d).
        site_i:     Left site index (0-based). Gate acts on (site_i, site_i+1).
        max_bond:   Maximum bond dimension after SVD truncation.
        svd_cutoff: Discard singular values smaller than this.
        absorb:     Where to absorb singular values after SVD split.
        normalize:  Renormalize the state after truncation.
        inplace:    If True, modify MPS tensors in place and return self.
                    If False (default), return a new MPS.

    Returns:
        (updated_mps, singular_values) where singular_values is the array of
        kept singular values at the bond between site_i and site_i+1.

    Raises:
        ValueError: If site_i is out of range or gate shape is incompatible.
    """
    L = mps.L
    j = site_i + 1

    if not (0 <= site_i < L - 1):
        raise ValueError(
            f"site_i={site_i} must satisfy 0 <= site_i < L-1={L - 1}."
        )

    # Physical dims at the two sites
    d_i = mps._physical_dims[site_i]
    d_j = mps._physical_dims[j]

    # Validate and reshape gate
    gate = np.asarray(gate)
    if gate.shape == (d_i * d_j, d_i * d_j):
        U = gate.reshape(d_i, d_j, d_i, d_j)
    elif gate.shape == (d_i, d_j, d_i, d_j):
        U = gate
    else:
        raise ValueError(
            f"Gate must have shape ({d_i*d_j},{d_i*d_j}) or "
            f"({d_i},{d_j},{d_i},{d_j}), got {gate.shape}."
        )

    # Retrieve raw tensor data
    A_i = mps.tensors[site_i].data   # (chiL, d_i, chi_mid)
    A_j = mps.tensors[j].data        # (chi_mid, d_j, chiR)

    if A_i is None or A_j is None:
        raise ValueError(
            f"MPS tensors at sites {site_i} and/or {j} have data=None "
            "(unmaterialized MPS)."
        )

    chiL    = A_i.shape[0]
    chi_mid = A_i.shape[2]
    chiR    = A_j.shape[2]

    # Contract two-site tensor: theta[a, i, j, b] = sum_c A_i[a,i,c] A_j[c,j,b]
    theta = np.einsum("aic,cjb->aijb", A_i, A_j)

    # Apply gate: theta'[a,i',j',b] = sum_{i,j} U[i',j',i,j] theta[a,i,j,b]
    theta_prime = np.einsum("mnij,aijb->amnb", U, theta)

    # SVD split
    A_new, S, B_new = _svd_split(
        theta=theta_prime,
        chiL=chiL,
        d=d_i,
        dR=d_j,
        chiR=chiR,
        max_bond=max_bond,
        svd_cutoff=svd_cutoff,
        absorb=absorb,
        normalize=normalize,
    )

    chi_new = A_new.shape[2]

    # Build updated tensors with corrected Index objects
    bond_left  = mps.bonds[site_i]           # unchanged
    phys_i     = mps.indices[site_i]         # unchanged
    phys_j     = mps.indices[j]              # unchanged
    bond_right = mps.bonds[j + 1]            # unchanged

    # New shared bond between site_i and j
    new_bond = Index(
        dim=chi_new,
        name=mps.bonds[j].name,
        tags=mps.bonds[j].tags,
    )

    new_tensor_i = Tensor(
        A_new.astype(mps.dtype, copy=False),
        indices=[bond_left, phys_i, new_bond],
    )
    new_tensor_j = Tensor(
        B_new.astype(mps.dtype, copy=False),
        indices=[new_bond, phys_j, bond_right],
    )

    if inplace:
        mps.tensors[site_i] = new_tensor_i
        mps.tensors[j]      = new_tensor_j
        mps.bonds[j]        = new_bond
        mps._bond_dims[j]   = chi_new
        return mps, S
    else:
        new_tensors = list(mps.tensors)
        new_tensors[site_i] = new_tensor_i
        new_tensors[j]      = new_tensor_j
        new_mps = MPS.from_tensors(new_tensors, name=mps.name)
        return new_mps, S