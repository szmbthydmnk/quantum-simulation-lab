"""
Finite-size nearest-neighbour TEBD (real-time evolution).

Index conventions
-----------------
MPS tensors : A[i] with shape (chiL, d, chiR)
Two-site gate U : shape (d*d, d*d) in the lexicographic basis
                  |s1 s2> = |s1> ⊗ |s2>.

Given an MPS and a set of nearest-neighbour gates on even and odd bonds,
TEBD applies a first-order Trotter step:

    U(dt) ≈ exp(-i H_even dt) exp(-i H_odd dt)

where H_even = sum over bonds (0,1), (2,3), ...
      H_odd  = sum over bonds (1,2), (3,4), ...

This module only knows about two-site gates; building those from a
Hamiltonian (e.g. XXZ, TFIM) is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[complex]]

@dataclass
class TEBDConfig:
    """
    Configuration for finite-size TEBD.

    Attributes
    ----------
    n_steps:
        Number of full TEBD steps to apply.
    normalize:
        If True, normalize the MPS after each full step. For exact
        unitary gates and no truncation this is unnecessary, but with
        truncation it can be helpful to counteract norm drift.
    verbose:
        Print simple step diagnostics (optional).
    """

    n_steps: int
    normalize: bool = True
    verbose: bool = False


def two_site_gate_from_hamiltonian(h_two_site: np.ndarray, 
                                   dt: complex,
                                   *,
                                   dtype: np.dtype = np.complex128,
                                   ) -> np.ndarray:
    """
    Build a two-site time-evolution gate

        U(dt) = exp(-i dt H_local)

    from a dense 2-site Hamiltonian H_local of shape (d^2, d^2).

    This uses an exact diagonalisation (eigh) and is intended for
    small local Hilbert spaces (d=2,3,...) only.
    """
    H = np.asarray(h_two_site, dtype=dtype)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            f"h_two_site must be a square matrix, got shape {H.shape!r}"
        )

    evals, evecs = np.linalg.eigh(H)
    phases = np.exp(-1j * dt * evals)
    U = (evecs * phases[None, :]) @ evecs.conj().T
    return U.astype(dtype, copy=False)


def _choose_chi(S: np.ndarray, truncation: TruncationPolicy | None) -> int:
    """Decide how many singular values to keep."""
    if truncation is None:
        return int(S.shape[0])
    return int(truncation.choose_bond_dim(S))


def apply_two_site_gate(mps: MPS,
                        gate: np.ndarray,
                        bond: int,
                        truncation: TruncationPolicy | None = None,
                        ) -> None:
    """
    Apply a two-site gate U on bond (bond, bond+1) of an MPS in-place.

    Parameters
    ----------
    mps:
        Input/output MPS, modified in-place.
    gate:
        Two-site gate U of shape (d^2, d^2) in the basis
        |s1 s2> = |s1> ⊗ |s2>.
    bond:
        Integer bond index i, meaning the gate acts on sites (i, i+1).
    truncation:
        Optional truncation policy for the SVD split on this bond.
        If None, keep full rank (no truncation).
    """
    
    L = len(mps)
    if not (0 <= bond < L - 1):
        raise ValueError(f"bond={bond} out of range for chain of length L={L}")

    A = mps.tensors[bond].data
    B = mps.tensors[bond + 1].data
    if A is None or B is None:
        raise ValueError("MPS tensors must be materialized (data not None).")

    chiL, d1, chiM = A.shape
    chiM2, d2, chiR = B.shape
    if chiM != chiM2:
        raise ValueError(
            f"Bond dimension mismatch at bond {bond}: "
            f"A.shape={A.shape}, B.shape={B.shape}"
        )
    if d1 != d2:
        raise ValueError(
            f"Physical dimension mismatch at bond {bond}: "
            f"A.shape={A.shape}, B.shape={B.shape}"
        )
    d = d1

    U = np.asarray(gate, dtype=mps.dtype)
    if U.shape != (d * d, d * d):
        raise ValueError(
            f"gate must have shape ({d*d}, {d*d}), got {U.shape!r}"
        )

    # Build 2-site tensor theta[a, s1, s2, c]
    theta = np.tensordot(A, B, axes=([2], [0]))  # (chiL,d,chiM) x (chiM,d,chiR)
    # -> (chiL, d, d, chiR)
    theta = theta.reshape(chiL, d * d, chiR)

    # Apply U on the physical legs: treat middle index as the d^2 space.
    # U[α,β] theta[a,β,c]  ->  tmp[α,a,c]
    tmp = np.tensordot(U, theta, axes=([1], [1]))  # (d^2, d^2) x (chiL,d^2,chiR)
    # tmp has shape (d^2, chiL, chiR) -> reshape back to (chiL,d,d,chiR)
    theta_new = np.transpose(tmp, (1, 0, 2)).reshape(chiL, d, d, chiR)

    # SVD splitting
    X = theta_new.reshape(chiL * d, d * chiR)
    Umat, S, Vh = np.linalg.svd(X, full_matrices=False)

    chi_keep = _choose_chi(S, truncation)
    chi_keep = max(1, min(chi_keep, S.size))

    Umat = Umat[:, :chi_keep]
    S = S[:chi_keep]
    Vh = Vh[:chi_keep, :]

    A_new = Umat.reshape(chiL, d, chi_keep)
    B_new = (S[:, None] * Vh).reshape(chi_keep, d, chiR)

    mps.tensors[bond].data = A_new.astype(mps.dtype, copy=False)
    mps.tensors[bond + 1].data = B_new.astype(mps.dtype, copy=False)


def _prepare_layer_gates(gates: Union[np.ndarray, Sequence[np.ndarray]],
                         L: int,
                         offset: int,
                         d: int,
                         ) -> List[Tuple[int, np.ndarray]]:
    """
    Prepare a list of (bond_index, gate) pairs for one TEBD layer.

    If `gates` is a single array of shape (d^2, d^2), it is broadcast to
    all bonds with the given parity (even/odd). If it is a sequence, it
    must have length equal to the number of such bonds.
    """
    
    bonds = list(range(offset, L - 1, 2))
    
    if isinstance(gates, np.ndarray):
        if gates.shape != (d * d, d * d):
            raise ValueError(
                f"Uniform gate must have shape ({d*d}, {d*d}), "
                f"got {gates.shape!r}"
            )
        return [(i, gates) for i in bonds]

    # Sequence of per-bond gates
    if len(gates) != len(bonds):
        raise ValueError(
            f"Expected {len(bonds)} gates for parity offset={offset}, "
            f"got {len(gates)}"
        )

    prepared: List[Tuple[int, np.ndarray]] = []
    for i, G in zip(bonds, gates):
        G_arr = np.asarray(G)
        if G_arr.shape != (d * d, d * d):
            raise ValueError(
                f"Gate on bond {i} must have shape ({d*d}, {d*d}), "
                f"got {G_arr.shape!r}"
            )
        prepared.append((i, G_arr))
    return prepared


def finite_tebd(mps0: MPS,
                gates_even: Union[np.ndarray, Sequence[np.ndarray]],
                gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
                config: TEBDConfig,
                truncation: TruncationPolicy | None = None,
                ) -> MPS:
    """
    Finite-size nearest-neighbour TEBD with first-order Trotter splitting.

        U(dt) ≈ exp(-i H_even dt) exp(-i H_odd dt)

    where H_even is the sum over even bonds and H_odd over odd bonds.
    The time step dt is encoded in the supplied two-site gates; this
    function does not construct them itself.

    Parameters
    ----------
    mps0:
        Initial MPS (not modified; a copy is evolved).
    gates_even:
        Two-site gate(s) for even bonds (0,1), (2,3), ...
        Either a single array of shape (d^2, d^2) (uniform coupling) or
        a sequence of such arrays of length equal to the number of even
        bonds.
    gates_odd:
        Same as `gates_even`, but for odd bonds (1,2), (3,4), ...
    config:
        TEBDConfig (number of steps, normalization, verbosity).
    truncation:
        Truncation policy applied at each SVD split. If None, no
        truncation (full rank) is used.

    Returns
    -------
    mps:
        Evolved MPS after `config.n_steps` Trotter steps.
    """
    
    if config.n_steps <= 0:
        raise ValueError("config.n_steps must be a positive integer")

    mps = mps0.copy()
    L = len(mps)
    
    if L < 2:
        raise ValueError("finite_tebd requires L >= 2")

    phys_dims = mps.physical_dims
    
    if len(set(phys_dims)) != 1:
        raise ValueError(
            f"finite_tebd currently assumes uniform physical dimension, got {phys_dims}"
        )
        
    d = phys_dims[0]

    layer_even = _prepare_layer_gates(gates_even, L=L, offset=0, d=d)
    layer_odd = _prepare_layer_gates(gates_odd, L=L, offset=1, d=d)

    for step in range(config.n_steps):
        # Even bonds
        for bond, G in layer_even:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        # Odd bonds
        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        if config.normalize:
            mps.normalize()

        if config.verbose:
            nrm = mps.norm()
            print(f"[finite_tebd] step {step+1}/{config.n_steps}, norm={nrm:.12f}")

    return mps

def finite_tebd_imaginary(mps0: MPS,
                          gates_even: Union[np.ndarray, Sequence[np.ndarray]],
                          gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
                          n_steps: int,
                          truncation: TruncationPolicy | None = None,
                          verbose: bool = False,
                          ) -> MPS:
    """
    Finite-size nearest-neighbour *imaginary-time* TEBD.

    This is a thin wrapper around :func:`finite_tebd` that interprets the
    supplied two-site gates as Euclidean evolution operators

        U(Δτ) = exp(-Δτ H_local),

    i.e. without the factor of -i. As these gates are non-unitary, the
    MPS is *always* normalized after each full step.

    Parameters
    ----------
    mps0:
        Initial MPS (not modified; a copy is evolved).
    gates_even:
        Two-site Euclidean gates for even bonds (0,1), (2,3), ...,
        either a single array of shape (d^2, d^2) or a sequence of such
        arrays matching the number of even bonds.
    gates_odd:
        Same as `gates_even`, but for odd bonds (1,2), (3,4), ...
    n_steps:
        Number of full imaginary-time TEBD steps to apply.
    truncation:
        Optional truncation policy used at each SVD split. If None, no
        truncation (full rank) is used.
    verbose:
        If True, log the MPS norm after each step.

    Returns
    -------
    mps:
        Evolved MPS after `n_steps` imaginary-time steps.
    """
    
    cfg = TEBDConfig(n_steps=n_steps, normalize=True, verbose=verbose)
    
    return finite_tebd(mps0=mps0,
                       gates_even=gates_even,
                       gates_odd=gates_odd,
                       config=cfg,
                       truncation=truncation)