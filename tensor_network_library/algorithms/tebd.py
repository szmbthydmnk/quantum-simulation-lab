# tensor_network_library/algorithms/tebd.py
"""
Finite-size nearest-neighbour TEBD (real- and imaginary-time evolution).

Index conventions
-----------------
MPS tensors : A[i] with shape (chiL, d, chiR)
Two-site gate U : shape (d*d, d*d) in the lexicographic basis
                  |s1 s2> = |s1> ⊗ |s2>.

First-order Trotter step:

    U₁(dt) ≈ exp(-i H_even dt) exp(-i H_odd dt)

Second-order (Strang) Trotter step:

    U₂(dt) = exp(-i H_even dt/2) exp(-i H_odd dt) exp(-i H_even dt/2)

where H_even = sum over bonds (0,1), (2,3), ...
      H_odd  = sum over bonds (1,2), (3,4), ...

This module only knows about two-site gates; building those from a
Hamiltonian (e.g. XXZ, TFIM) is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[complex]]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gate construction
# ---------------------------------------------------------------------------


def two_site_gate_from_hamiltonian(
    h_two_site: np.ndarray,
    dt: complex,
    *,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Build a two-site time-evolution gate

        U(dt) = exp(-i dt H_local)

    from a dense 2-site Hamiltonian H_local of shape (d^2, d^2).

    Uses exact diagonalisation (eigh) and is intended for small local
    Hilbert spaces (d = 2, 3, ...) only.
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


def two_site_gate_imaginary(
    h_two_site: np.ndarray,
    dtau: float,
    *,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Build a two-site imaginary-time evolution operator

        U(dtau) = exp(-dtau H_local)

    from a dense 2-site Hamiltonian H_local of shape (d^2, d^2).

    The result is Hermitian and positive semi-definite for dtau > 0.
    """
    H = np.asarray(h_two_site, dtype=dtype)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(
            f"h_two_site must be a square matrix, got shape {H.shape!r}"
        )

    evals, evecs = np.linalg.eigh(H)
    factors = np.exp(-dtau * evals)
    U = (evecs * factors[None, :]) @ evecs.conj().T
    return U.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _choose_chi(S: np.ndarray, truncation: TruncationPolicy | None) -> int:
    """Decide how many singular values to keep."""
    if truncation is None:
        return int(S.shape[0])
    return int(truncation.choose_bond_dim(S))


def _prepare_layer_gates(
    gates: Union[np.ndarray, Sequence[np.ndarray]],
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


# ---------------------------------------------------------------------------
# Two-site gate application
# ---------------------------------------------------------------------------


def apply_two_site_gate(
    mps: MPS,
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

    # Build 2-site tensor theta[chiL, d, d, chiR]
    theta = np.tensordot(A, B, axes=([2], [0]))   # (chiL,d,chiM)x(chiM,d,chiR) -> (chiL,d,d,chiR)
    theta = theta.reshape(chiL, d * d, chiR)

    # Apply gate: U[α,β] theta[a,β,c] -> tmp[α,a,c]
    tmp = np.tensordot(U, theta, axes=([1], [1]))  # (d^2,d^2)x(chiL,d^2,chiR) -> (d^2,chiL,chiR)
    theta_new = np.transpose(tmp, (1, 0, 2)).reshape(chiL, d, d, chiR)

    # SVD split
    X = theta_new.reshape(chiL * d, d * chiR)
    Umat, S, Vh = np.linalg.svd(X, full_matrices=False)

    chi_keep = _choose_chi(S, truncation)
    chi_keep = max(1, min(chi_keep, S.size))

    Umat = Umat[:, :chi_keep]
    S = S[:chi_keep]
    Vh = Vh[:chi_keep, :]

    # Absorb singular values into the right tensor (right-canonical sweep)
    A_new = Umat.reshape(chiL, d, chi_keep)
    B_new = (S[:, None] * Vh).reshape(chi_keep, d, chiR)

    mps.tensors[bond].data = A_new.astype(mps.dtype, copy=False)
    mps.tensors[bond + 1].data = B_new.astype(mps.dtype, copy=False)


# ---------------------------------------------------------------------------
# Public TEBD sweepers
# ---------------------------------------------------------------------------


def finite_tebd(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    config: TEBDConfig,
    truncation: TruncationPolicy | None = None,
) -> MPS:
    """
    Finite-size nearest-neighbour TEBD with first-order Trotter splitting.

        U₁(dt) ≈ exp(-i H_even dt) exp(-i H_odd dt)

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
        for bond, G in layer_even:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        if config.normalize:
            mps.normalize()

        if config.verbose:
            nrm = mps.norm()
            print(f"[finite_tebd] step {step+1}/{config.n_steps}, norm={nrm:.12f}")

    return mps


def finite_tebd_strang(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_even_half: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    config: TEBDConfig,
    truncation: TruncationPolicy | None = None,
) -> MPS:
    """
    Finite-size nearest-neighbour TEBD with second-order (Strang) Trotter.

        U₂(dt) = exp(-i H_even dt/2) exp(-i H_odd dt) exp(-i H_even dt/2)

    The caller is responsible for supplying both the full-step even gates
    (``gates_even``) and the half-step even gates (``gates_even_half``).
    A typical construction:

        G_even_half = two_site_gate_from_hamiltonian(h, dt / 2)
        G_even_full = two_site_gate_from_hamiltonian(h, dt)
        G_odd_full  = two_site_gate_from_hamiltonian(h, dt)

    Note: the final half-step of step n and the initial half-step of
    step n+1 can be merged into a single full even-bond sweep.  This
    optimisation is *not* applied here to keep the code readable; it
    reduces the number of SVD calls by roughly 1/3 for long runs.

    Parameters
    ----------
    mps0:
        Initial MPS (not modified; a copy is evolved).
    gates_even:
        Full-step two-site gate(s) for even bonds. Used as the middle
        even layer when consecutive steps are chained (currently unused
        internally — reserved for a future merged-step variant).
    gates_even_half:
        Half-step two-site gate(s) for even bonds.  Applied at the start
        and end of every Trotter step.
    gates_odd:
        Full-step two-site gate(s) for odd bonds.
    config:
        TEBDConfig (number of steps, normalization, verbosity).
    truncation:
        Truncation policy applied at each SVD split.

    Returns
    -------
    mps:
        Evolved MPS after `config.n_steps` second-order Trotter steps.
    """
    if config.n_steps <= 0:
        raise ValueError("config.n_steps must be a positive integer")

    mps = mps0.copy()
    L = len(mps)

    if L < 2:
        raise ValueError("finite_tebd_strang requires L >= 2")

    phys_dims = mps.physical_dims
    if len(set(phys_dims)) != 1:
        raise ValueError(
            f"finite_tebd_strang currently assumes uniform physical dimension, "
            f"got {phys_dims}"
        )
    d = phys_dims[0]

    layer_even_half = _prepare_layer_gates(gates_even_half, L=L, offset=0, d=d)
    layer_odd = _prepare_layer_gates(gates_odd, L=L, offset=1, d=d)

    for step in range(config.n_steps):
        # dt/2 on even bonds
        for bond, G in layer_even_half:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        # dt on odd bonds
        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        # dt/2 on even bonds
        for bond, G in layer_even_half:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        if config.normalize:
            mps.normalize()

        if config.verbose:
            nrm = mps.norm()
            print(
                f"[finite_tebd_strang] step {step+1}/{config.n_steps}, "
                f"norm={nrm:.12f}"
            )

    return mps


def finite_tebd_imaginary(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    n_steps: int,
    truncation: TruncationPolicy | None = None,
    verbose: bool = False,
) -> MPS:
    """
    Finite-size nearest-neighbour imaginary-time TEBD.

    Thin wrapper around :func:`finite_tebd` that interprets the supplied
    gates as Euclidean evolution operators

        U(Δτ) = exp(-Δτ H_local),

    i.e. without the factor of -i.  The MPS is always normalised after
    each full step because these gates are non-unitary.

    Parameters
    ----------
    mps0:
        Initial MPS (not modified; a copy is evolved).
    gates_even:
        Euclidean two-site gate(s) for even bonds.
    gates_odd:
        Euclidean two-site gate(s) for odd bonds.
    n_steps:
        Number of full imaginary-time steps.
    truncation:
        Optional truncation policy.
    verbose:
        If True, log the MPS norm after each step.

    Returns
    -------
    mps:
        Evolved MPS after `n_steps` imaginary-time steps.
    """
    cfg = TEBDConfig(n_steps=n_steps, normalize=True, verbose=verbose)
    return finite_tebd(
        mps0=mps0,
        gates_even=gates_even,
        gates_odd=gates_odd,
        config=cfg,
        truncation=truncation,
    )


# ---------------------------------------------------------------------------
# Observable measurement
# ---------------------------------------------------------------------------


def measure_local(
    mps: MPS,
    ops: Union[np.ndarray, Sequence[np.ndarray]],
) -> np.ndarray:
    """
    Compute single-site expectation values <ψ|O_i|ψ> for all sites i.

    Uses a left-right transfer matrix sweep. Does not require the MPS to
    be in any particular canonical form; the result is divided by <ψ|ψ>.
    Cost: O(L χ² d).

    Parameters
    ----------
    mps:
        The MPS |ψ⟩.
    ops:
        Either a single operator of shape (d, d) applied uniformly to
        all sites, or a sequence of L operators of shape (d, d).

    Returns
    -------
    exp_vals : np.ndarray, shape (L,)
        Real part of <O_i> for each site i.
    """
    L = len(mps)
    phys_dims = mps.physical_dims
    if len(set(phys_dims)) != 1:
        raise ValueError(
            "measure_local currently requires a uniform physical dimension."
        )
    d = phys_dims[0]

    # Build per-site operator list
    if isinstance(ops, np.ndarray) and ops.ndim == 2:
        op_list: List[np.ndarray] = [np.asarray(ops, dtype=mps.dtype)] * L
    else:
        ops_seq = list(ops)
        if len(ops_seq) != L:
            raise ValueError(
                f"ops sequence must have length L={L}, got {len(ops_seq)}"
            )
        op_list = [np.asarray(o, dtype=mps.dtype) for o in ops_seq]

    for i, O in enumerate(op_list):
        if O.shape != (d, d):
            raise ValueError(
                f"Operator at site {i} must have shape ({d},{d}), got {O.shape}"
            )

    tensors = [mps.tensors[i].data for i in range(L)]
    for i, t in enumerate(tensors):
        if t is None:
            raise ValueError(f"MPS tensor at site {i} is not materialized.")

    # ------------------------------------------------------------------
    # Pass 1: accumulate left environments L_env[i] = transfer matrix
    #         of sites 0..i-1.  Shape (chiL_i, chiL_i).
    # L_env[0] = identity (nothing to the left of site 0).
    # ------------------------------------------------------------------
    left_envs: List[np.ndarray] = []
    env = np.eye(tensors[0].shape[0], dtype=mps.dtype)
    for i in range(L):
        left_envs.append(env.copy())
        A = tensors[i]                               # (chiL, d, chiR)
        env = np.einsum("ab,asc,bsd->cd", env, A, A.conj())
    # env is now the full norm-squared matrix; trace gives <ψ|ψ>
    norm_sq = float(np.real(np.trace(env)))

    # ------------------------------------------------------------------
    # Pass 2: accumulate right environments R_env[i] = transfer matrix
    #         of sites i+1..L-1.  Shape (chiR_i, chiR_i).
    # R_env[L-1] = identity (nothing to the right of the last site).
    # ------------------------------------------------------------------
    right_envs: List[np.ndarray] = [None] * L   # type: ignore[list-item]
    env_r = np.eye(tensors[-1].shape[2], dtype=mps.dtype)
    right_envs[L - 1] = env_r.copy()
    for i in range(L - 2, -1, -1):
        A = tensors[i + 1]                           # (chiL, d, chiR)
        env_r = np.einsum("asc,cd,bsd->ab", A, env_r, A.conj())
        right_envs[i] = env_r.copy()

    # ------------------------------------------------------------------
    # Pass 3: sandwich each site with its operator.
    # <O_i> (unnormalized) =
    #   sum_{a,b,s,t,c,d} L[a,b] A[a,s,c] O[s,t] A*[b,t,d] R[c,d]
    # ------------------------------------------------------------------
    exp_vals = np.zeros(L, dtype=complex)
    for i in range(L):
        A = tensors[i]                               # (chiL, d, chiR)
        O = op_list[i]                               # (d, d)
        L_env = left_envs[i]                         # (chiL, chiL)
        R_env = right_envs[i]                        # (chiR, chiR)
        val = np.einsum("ab,asc,st,btd,cd->", L_env, A, O, A.conj(), R_env)
        exp_vals[i] = val

    exp_vals /= norm_sq

    imag_max = float(np.max(np.abs(exp_vals.imag)))
    if imag_max > 1e-10:
        import warnings
        warnings.warn(
            f"measure_local: largest imaginary part = {imag_max:.3e}; "
            "operator may not be Hermitian or MPS has numerical noise.",
            stacklevel=2,
        )

    return exp_vals.real