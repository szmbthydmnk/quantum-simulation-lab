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

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.tensor import Tensor
from tensor_network_library.core.index import Index

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
# Result
# ---------------------------------------------------------------------------


@dataclass
class TEBDResult:
    """
    Return object for all finite-size TEBD sweepers.

    Attributes
    ----------
    mps:
        The evolved MPS after all Trotter steps.
    n_steps:
        Number of full Trotter steps actually performed.
    norm_history:
        MPS norm recorded *after* each full step (before any explicit
        re-normalisation). For unitary evolution without truncation this
        should remain ≈ 1; for imaginary-time evolution it tracks the
        raw norm decay before the in-loop normalisation is applied.
    energy_history:
        Optional per-step energy estimates.  Populated only when a
        ``measure_fn`` is passed to :func:`finite_tebd_imaginary`;
        empty list otherwise.
    """

    mps: MPS
    n_steps: int
    norm_history: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)


@dataclass
class GroundStateResult:
    """
    Return object for :func:`ground_state_search`.

    Attributes
    ----------
    mps:
        Ground-state MPS estimate after convergence or ``max_steps``.
    energy_history:
        Per-step energy estimate ``<ψ|H|ψ>`` evaluated after each
        normalised imaginary-time step.
    norm_history:
        Raw MPS norm before normalisation at each step (tracks the
        exponential decay curve of imaginary-time evolution).
    converged:
        True if ``|E_n - E_{n-1}| < energy_tol`` was satisfied before
        ``max_steps`` was reached.
    n_steps:
        Number of imaginary-time steps actually performed.
    """

    mps: MPS
    energy_history: List[float]
    norm_history: List[float]
    converged: bool
    n_steps: int
    

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


def _left_canonicalize_inplace(mps: MPS) -> None:
    """
    Bring an MPS into left-canonical form via a left-to-right QR sweep.

    Each site tensor A[i] of shape (chiL, d, chiR) is reshaped to
    (chiL*d, chiR), QR-decomposed, and the R factor is absorbed into
    A[i+1].  The final site is left as-is (it carries the norm).

    This is O(L chi^2 d) and is used to ensure Im(<H>) ≈ 0 when calling
    measure_bond_energies on a mixed-gauge MPS after a TEBD sweep.
    """
    L = len(mps)
    for i in range(L - 1):
        A = mps.tensors[i].data
        chiL, d, chiR = A.shape
        # Reshape and QR
        M = A.reshape(chiL * d, chiR)
        Q, R = np.linalg.qr(M)
        chi_new = Q.shape[1]
        A_new = Q.reshape(chiL, d, chi_new)
        # Absorb R into next site
        B = mps.tensors[i + 1].data
        chiM, d_next, chiR_next = B.shape
        B_new = np.tensordot(R, B, axes=([1], [0]))  # (chi_new, d_next, chiR_next)
        # Update bond index
        old_bond = mps.bonds[i + 1]
        new_bond = Index(dim=chi_new, name=old_bond.name, tags=old_bond.tags)
        mps.tensors[i] = Tensor(
            A_new.astype(mps.dtype, copy=False),
            indices=[mps.bonds[i], mps.indices[i], new_bond],
        )
        mps.tensors[i + 1] = Tensor(
            B_new.astype(mps.dtype, copy=False),
            indices=[new_bond, mps.indices[i + 1], mps.bonds[i + 2]],
        )
        mps.bonds[i + 1] = new_bond
        mps._bond_dims[i + 1] = chi_new


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

    # Build a fresh shared bond Index with the updated dimension.
    # Reusing the old Index (dim may have changed) would leave stale
    # dim metadata on the Tensor objects and break norm() / conj().
    old_bond = mps.bonds[bond + 1]
    new_bond = Index(
        dim=chi_keep,
        name=old_bond.name,
        tags=old_bond.tags,
    )

    # Replace both site tensors with fresh Tensor objects that carry the
    # correct indices — do NOT mutate .data in-place, as that leaves the
    # existing Index objects with wrong dims.
    mps.tensors[bond] = Tensor(
        A_new.astype(mps.dtype, copy=False),
        indices=[mps.bonds[bond], mps.indices[bond], new_bond],
    )
    mps.tensors[bond + 1] = Tensor(
        B_new.astype(mps.dtype, copy=False),
        indices=[new_bond, mps.indices[bond + 1], mps.bonds[bond + 2]],
    )

    # Keep the MPS bookkeeping consistent.
    mps.bonds[bond + 1] = new_bond
    mps._bond_dims[bond + 1] = chi_keep


# ---------------------------------------------------------------------------
# Public TEBD sweepers
# ---------------------------------------------------------------------------


def finite_tebd(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    config: TEBDConfig,
    truncation: TruncationPolicy | None = None,
) -> TEBDResult:
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
    result : TEBDResult
        ``result.mps`` is the evolved MPS after ``config.n_steps`` Trotter
        steps. ``result.norm_history`` contains the MPS norm recorded after
        each full step (before any explicit re-normalisation).
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

    norm_history: List[float] = []

    for step in range(config.n_steps):
        for bond, G in layer_even:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        nrm = mps.norm()
        norm_history.append(float(nrm))

        if config.normalize:
            mps.normalize()

        if config.verbose:
            print(f"[finite_tebd] step {step+1}/{config.n_steps}, norm={nrm:.12f}")

    return TEBDResult(mps=mps, n_steps=config.n_steps, norm_history=norm_history)


def finite_tebd_strang(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_even_half: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    config: TEBDConfig,
    truncation: TruncationPolicy | None = None,
) -> TEBDResult:
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
    result : TEBDResult
        ``result.mps`` is the evolved MPS after ``config.n_steps``
        second-order Trotter steps. ``result.norm_history`` contains the
        MPS norm recorded after each full step.
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

    norm_history: List[float] = []

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

        nrm = mps.norm()
        norm_history.append(float(nrm))

        if config.normalize:
            mps.normalize()

        if config.verbose:
            print(
                f"[finite_tebd_strang] step {step+1}/{config.n_steps}, "
                f"norm={nrm:.12f}"
            )

    return TEBDResult(mps=mps, n_steps=config.n_steps, norm_history=norm_history)


def finite_tebd_imaginary(
    mps0: MPS,
    gates_even: Union[np.ndarray, Sequence[np.ndarray]],
    gates_odd: Union[np.ndarray, Sequence[np.ndarray]],
    n_steps: int,
    truncation: TruncationPolicy | None = None,
    verbose: bool = False,
    measure_fn: Optional[Callable[[MPS], float]] = None,
) -> TEBDResult:
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
        If True, log the MPS norm (and energy, if measure_fn is given)
        after each step.
    measure_fn:
        Optional callable ``(mps: MPS) -> float`` invoked on the
        (normalised) MPS after every step.  Intended for in-loop energy
        tracking; the returned values are stored in
        ``TEBDResult.energy_history``.

        Example::

            def energy(mps):
                return float(np.sum(measure_bond_energies(mps, H_local)))

            result = finite_tebd_imaginary(..., measure_fn=energy)
            plt.plot(result.energy_history)

    Returns
    -------
    result : TEBDResult
        ``result.mps`` is the evolved MPS. ``result.norm_history``
        records the raw norm *before* renormalisation at each step.
        ``result.energy_history`` is populated when ``measure_fn`` is
        provided; empty list otherwise.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")

    mps = mps0.copy()
    L = len(mps)

    if L < 2:
        raise ValueError("finite_tebd_imaginary requires L >= 2")

    phys_dims = mps.physical_dims
    if len(set(phys_dims)) != 1:
        raise ValueError(
            f"finite_tebd_imaginary currently assumes uniform physical dimension, "
            f"got {phys_dims}"
        )
    d = phys_dims[0]

    layer_even = _prepare_layer_gates(gates_even, L=L, offset=0, d=d)
    layer_odd = _prepare_layer_gates(gates_odd, L=L, offset=1, d=d)

    norm_history: List[float] = []
    energy_history: List[float] = []

    for step in range(n_steps):
        for bond, G in layer_even:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        nrm = mps.norm()
        norm_history.append(float(nrm))

        # Always normalise: imaginary-time gates are non-unitary
        mps.normalize()

        if measure_fn is not None:
            energy = measure_fn(mps)
            energy_history.append(float(energy))

        if verbose:
            msg = f"[finite_tebd_imaginary] step {step+1}/{n_steps}, norm={nrm:.12f}"
            if measure_fn is not None:
                msg += f", energy={energy_history[-1]:.12f}"
            print(msg)

    return TEBDResult(
        mps=mps,
        n_steps=n_steps,
        norm_history=norm_history,
        energy_history=energy_history,
    )


def measure_bond_energies(
    mps: MPS,
    h_bonds: Union[np.ndarray, Sequence[np.ndarray]],
) -> np.ndarray:
    """
    Compute two-site bond energy expectations <ψ|h_{i,i+1}|ψ> for all bonds.

    Parameters
    ----------
    mps:
        The MPS |ψ⟩.  Need not be normalised; result is divided by <ψ|ψ>.
    h_bonds:
        Local two-site Hamiltonian(s) of shape ``(d^2, d^2)``.
        Either a single uniform array or a sequence of ``L-1`` arrays.

    Returns
    -------
    energies : np.ndarray, shape (L-1,)
        Real part of <h_{i,i+1}> for each bond i.
    """
    L = len(mps)
    if L < 2:
        raise ValueError("measure_bond_energies requires L >= 2")

    phys_dims = mps.physical_dims
    if len(set(phys_dims)) != 1:
        raise ValueError(
            "measure_bond_energies currently assumes uniform physical dimension, "
            f"got {phys_dims}"
        )
    d = phys_dims[0]

    if isinstance(h_bonds, np.ndarray) and h_bonds.ndim == 2:
        h_list: List[np.ndarray] = [np.asarray(h_bonds, dtype=mps.dtype)] * (L - 1)
    else:
        h_list = [np.asarray(h, dtype=mps.dtype) for h in h_bonds]
        if len(h_list) != L - 1:
            raise ValueError(
                f"h_bonds sequence must have length L-1={L-1}, got {len(h_list)}"
            )

    for i, h in enumerate(h_list):
        if h.shape != (d * d, d * d):
            raise ValueError(
                f"h_bonds[{i}] must have shape ({d*d},{d*d}), got {h.shape}"
            )

    tensors = [mps.tensors[i].data for i in range(L)]
    for i, t in enumerate(tensors):
        if t is None:
            raise ValueError(f"MPS tensor at site {i} is not materialized.")

    # Norm squared via full transfer matrix sweep
    env = np.eye(tensors[0].shape[0], dtype=np.complex128)
    for A in tensors:
        env = np.einsum("ab,asc,bsd->cd", env, A, A.conj())
    norm_sq = float(np.real(np.trace(env)))

    # Left environments: left_envs[i] has shape (chi_i, chi_i)
    left_envs: List[np.ndarray] = []
    env = np.eye(tensors[0].shape[0], dtype=np.complex128)
    for A in tensors:
        left_envs.append(env.copy())
        env = np.einsum("ab,asc,bsd->cd", env, A, A.conj())

    # Right environments: right_envs[i] has shape (chi_{i+1}, chi_{i+1})
    right_envs: List[np.ndarray] = [None] * L  # type: ignore[list-item]
    env_r = np.eye(tensors[-1].shape[2], dtype=np.complex128)
    right_envs[L - 1] = env_r.copy()
    for i in range(L - 2, -1, -1):
        A = tensors[i + 1]
        env_r = np.einsum("asc,cd,bsd->ab", A, env_r, A.conj())
        right_envs[i] = env_r.copy()

    # Two-site sandwich: <ψ|h_{i,i+1}|ψ> =
    #   L[a,b] A_i[a,s,c] A_{i+1}[c,t,e] h[s't',st] A_i*[b,s',d] A_{i+1}*[d,t',f] R[e,f]
    energies = np.zeros(L - 1, dtype=complex)
    for i in range(L - 1):
        Ai  = tensors[i]          # (chiL, d, chiM)
        Aj  = tensors[i + 1]      # (chiM, d, chiR)
        L_e = left_envs[i]        # (chiL, chiL)
        R_e = right_envs[i + 1]   # (chiR, chiR)
        h   = h_list[i].reshape(d, d, d, d)  # (s', t', s, t)

        val = np.einsum(
            "ab,asc,ctf,mnst,bmd,dnf,ef->",
            L_e, Ai, Aj, h, Ai.conj(), Aj.conj(), R_e,
        )
        energies[i] = val

    energies /= norm_sq

    imag_max = float(np.max(np.abs(energies.imag)))
    if imag_max > 1e-10:
        import warnings
        warnings.warn(
            f"measure_bond_energies: largest imaginary part = {imag_max:.3e}; "
            "Hamiltonian may not be Hermitian or MPS has numerical noise.",
            stacklevel=2,
        )

    return energies.real


def ground_state_search(
    mps0: MPS,
    h_bonds: Union[np.ndarray, Sequence[np.ndarray]],
    *,
    dtau: float = 0.05,
    max_steps: int = 500,
    chi_max: int = 32,
    svd_cutoff: float = 1e-12,
    energy_tol: float = 1e-8,
    verbose: bool = False,
) -> GroundStateResult:
    """
    Ground-state search via imaginary-time TEBD.

    Applies repeated first-order Trotter steps of

        |ψ_{n+1}⟩ ∝ exp(-Δτ H) |ψ_n⟩

    renormalising after every step, until the per-step energy change
    ``|E_n - E_{n-1}|`` falls below ``energy_tol`` or ``max_steps``
    is exhausted.

    The energy at step n is estimated as

        E_n = Σ_i <ψ_n | h_{i,i+1} | ψ_n>

    using a two-site sandwich via :func:`measure_bond_energies` on a
    left-canonical copy of the MPS (ensuring Im(E) ≈ 0).

    Parameters
    ----------
    mps0:
        Initial MPS (not modified; a copy is evolved).  A random MPS
        (:meth:`MPS.from_random`) with sufficient bond dimension works
        well.  The MPS is normalised internally before the first step.
    h_bonds:
        Local two-site Hamiltonian(s) of shape ``(d^2, d^2)``.
        Either a single array (uniform coupling, same for every bond)
        or a sequence of ``L-1`` arrays for inhomogeneous systems.
    dtau:
        Imaginary time step Δτ.  Smaller values give better Trotter
        accuracy at the cost of more steps.  Typical range: 0.01–0.1.
    max_steps:
        Hard upper bound on the number of imaginary-time steps.
    chi_max:
        Maximum bond dimension kept after each SVD truncation.
    svd_cutoff:
        Singular values whose *square* is below this threshold are
        discarded (passed to :class:`TruncationPolicy`).
    energy_tol:
        Convergence criterion: stop when
        ``|E_n - E_{n-1}| < energy_tol``.
    verbose:
        If True, print step diagnostics.

    Returns
    -------
    GroundStateResult
        See :class:`GroundStateResult` for field descriptions.

    Notes
    -----
    * The Trotter splitting is first-order (even bonds then odd bonds).
      For high accuracy, reduce ``dtau`` and increase ``max_steps``,
      or switch to a Strang-split schedule externally.
    * Bond dimension grows from the initial MPS up to ``chi_max``.
      Initialising with a small random MPS and a generous ``chi_max``
      is the standard approach.
    * Energy convergence does **not** imply that the Trotter error
      is negligible; always verify by decreasing ``dtau``.
    * Energy is measured on a left-canonical copy of the MPS so that
      Im(<H>) is at machine precision; the evolved MPS itself is not
      modified by the measurement.
    """
    if dtau <= 0.0:
        raise ValueError(f"dtau must be positive, got {dtau}")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if chi_max <= 0:
        raise ValueError(f"chi_max must be positive, got {chi_max}")
    if energy_tol <= 0.0:
        raise ValueError(f"energy_tol must be positive, got {energy_tol}")

    L = len(mps0)
    if L < 2:
        raise ValueError("ground_state_search requires L >= 2")

    phys_dims = mps0.physical_dims
    if len(set(phys_dims)) != 1:
        raise ValueError(
            "ground_state_search currently assumes uniform physical dimension, "
            f"got {phys_dims}"
        )
    d = phys_dims[0]

    # --- Build imaginary-time gates ---
    # Uniform h_bonds: broadcast; per-bond: validate length
    if isinstance(h_bonds, np.ndarray) and h_bonds.ndim == 2:
        h_list: List[np.ndarray] = [np.asarray(h_bonds, dtype=np.complex128)] * (L - 1)
    else:
        h_list = [np.asarray(h, dtype=np.complex128) for h in h_bonds]
        if len(h_list) != L - 1:
            raise ValueError(
                f"h_bonds sequence must have length L-1={L-1}, got {len(h_list)}"
            )

    gates = [two_site_gate_imaginary(h, dtau) for h in h_list]
    gates_even = [gates[i] for i in range(0, L - 1, 2)]
    gates_odd  = [gates[i] for i in range(1, L - 1, 2)]

    truncation = TruncationPolicy(max_bond_dim=chi_max, cutoff=svd_cutoff)

    # --- Energy measurement on a left-canonical copy ---
    # After a TEBD sweep the MPS is in a mixed gauge (S absorbed into the
    # right tensor of the last updated bond).  Calling measure_bond_energies
    # directly on this state produces non-identity transfer matrices and a
    # large Im(<H>).  We instead QR-sweep a *copy* to left-canonical form
    # before every measurement; the evolved MPS itself is untouched.
    def _energy(mps: MPS) -> float:
        mps_lc = mps.copy()
        _left_canonicalize_inplace(mps_lc)
        return float(np.sum(measure_bond_energies(mps_lc, h_list)))

    # --- Evolve ---
    mps = mps0.copy()
    mps.normalize()

    norm_history: List[float] = []
    energy_history: List[float] = []
    converged = False

    layer_even = _prepare_layer_gates(gates_even, L=L, offset=0, d=d)
    layer_odd  = _prepare_layer_gates(gates_odd,  L=L, offset=1, d=d)

    for step in range(max_steps):
        for bond, G in layer_even:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)
        for bond, G in layer_odd:
            apply_two_site_gate(mps, G, bond=bond, truncation=truncation)

        nrm = mps.norm()
        norm_history.append(float(nrm))
        mps.normalize()

        energy = _energy(mps)
        energy_history.append(energy)

        if verbose:
            print(
                f"[ground_state_search] step {step+1}/{max_steps}  "
                f"norm={nrm:.10f}  E={energy:.12f}"
            )

        # Convergence check: need at least two energy measurements
        if len(energy_history) >= 2:
            if abs(energy_history[-1] - energy_history[-2]) < energy_tol:
                converged = True
                if verbose:
                    print(
                        f"[ground_state_search] converged at step {step+1}  "
                        f"ΔE={abs(energy_history[-1]-energy_history[-2]):.3e}"
                    )
                break

    return GroundStateResult(
        mps=mps,
        energy_history=energy_history,
        norm_history=norm_history,
        converged=converged,
        n_steps=len(energy_history),
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
    # R_env[L-1] = identity (nothing to the right of the last site)
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
