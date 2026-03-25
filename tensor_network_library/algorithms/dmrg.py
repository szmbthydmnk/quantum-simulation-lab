"""DMRG algorithms.

This module implements finite-size DMRG on top of the core
MPS/MPO/Environment infrastructure.

The implementation is intentionally explicit and heavily documented to
serve both as a reference and as a starting point for more advanced
variants (two-site updates, mixed canonical forms, etc.).

High-level design
-----------------

* The public API is a single entry point :func:`finite_dmrg` which takes
  an :class:`Environment`, an :class:`MPO`, and an initial :class:`MPS`,
  together with a small :class:`DMRGConfig` object.
* The algorithm performs 1-site DMRG sweeps, updating one MPS tensor at a
  time using a local effective Hamiltonian built from left/right
  environments and the MPO.
* Local effective Hamiltonians are constructed as dense matrices on the
  virtual+physical Hilbert space of a single site. This keeps the code
  compact and easy to reason about; for moderate bond dimensions the
  cubic cost in the local dimension is acceptable.
* Energies are evaluated using the efficient environment-based helper
  :func:`tensor_network_library.core.utils.expectation_value_env`.

The current implementation keeps the bond dimensions fixed (determined
by the initial MPS). This is the natural behaviour for 1-site DMRG and
is sufficient to demonstrate convergence on the simple Hamiltonians in
``examples.dmrg_hamiltonians``. A 2-site variant with dynamic
truncation can be added later without changing the public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.utils import (
    expectation_value_env,
    build_left_environments,
    build_right_environments,
)


@dataclass
class DMRGConfig:
    """Configuration parameters for finite-size DMRG.

    Attributes
    ----------
    max_sweeps:
        Maximum number of full sweeps (left-to-right plus right-to-left)
        to perform.
    energy_tol:
        Convergence tolerance on the change in energy between sweeps.
        When successive energies differ by less than this value, the
        algorithm stops.
    verbose:
        If ``True``, the implementation may emit diagnostic information
        to stdout (primarily for examples / notebooks).
    """

    max_sweeps: int = 10
    energy_tol: float = 1e-8
    verbose: bool = False


@dataclass
class DMRGResult:
    """Result container for a finite-size DMRG run.

    Attributes
    ----------
    mps:
        Final MPS approximation to the ground state.
    energies:
        List of energy estimates recorded after each full sweep.
    bond_dims:
        Snapshot of MPS bond dimensions after each sweep. Each entry is
        the list ``mps.bond_dims`` for that sweep.
    """

    mps: MPS
    energies: List[float]
    bond_dims: List[List[int]]


def _build_local_heff(L_i: np.ndarray, W_i: np.ndarray, R_ip1: np.ndarray) -> np.ndarray:
    """Construct the dense 1-site effective Hamiltonian H_eff at site i.

    The effective Hamiltonian acts on the virtual+physical space of a
    single site, with composite indices (l, s, r) where ``l`` and ``r``
    are the left/right bond indices of the MPS tensor and ``s`` is the
    physical index.

    Given

    - L_i(a, b, x): left environment at bond i,
    - W_i(x, i, j, y): MPO tensor at site i,
    - R_{i+1}(c, d, y): right environment at bond i+1,

    the matrix elements of H_eff are

        H_eff[(a, s, c), (b, t, d)] =
            sum_{x, y} L_i(a, b, x) * W_i(x, s, t, y) * R_{i+1}(c, d, y).

    This is implemented as a single ``einsum`` followed by a reshape.

    Parameters
    ----------
    L_i:
        Left environment at bond i, shape ``(chiL, chiL, wL)``.
    W_i:
        MPO tensor at site i, shape ``(wL, d, d, wR)``.
    R_ip1:
        Right environment at bond i+1, shape ``(chiR, chiR, wR)``.

    Returns
    -------
    H_eff:
        Dense effective Hamiltonian matrix of shape
        ``(chiL * d * chiR, chiL * d * chiR)``.
    """

    # H_tensor[a, s, c, b, t, d] = sum_{x,y} L_i[a,b,x] * W_i[x,s,t,y] * R_{i+1}[c,d,y]
    H_tensor = np.einsum("abx,xijy,cdy->aicbjd", L_i, W_i, R_ip1, optimize=True)

    chiL, d, chiR, chiL2, d2, chiR2 = H_tensor.shape
    assert chiL == chiL2 and chiR == chiR2 and d == d2

    dim = chiL * d * chiR
    H_eff = H_tensor.reshape(dim, dim)
    return H_eff


def _optimize_site(L_i: np.ndarray, W_i: np.ndarray, R_ip1: np.ndarray, A_i: np.ndarray) -> np.ndarray:
    """Optimize a single MPS tensor at site i.

    This function solves the local eigenproblem

        H_eff |theta> = E |theta>

    where H_eff is the 1-site effective Hamiltonian constructed from the
    left/right environments and the MPO tensor at site i, and |theta>
    is the flattened MPS tensor ``A_i``. The smallest eigenvector is
    reshaped back to the original tensor shape and returned.

    Parameters
    ----------
    L_i, W_i, R_ip1:
        Left/right environments and MPO tensor as in
        :func:`_build_local_heff`.
    A_i:
        Current MPS tensor at site i, shape ``(chiL, d, chiR)``. Only its
        shape is used; the initial vector is not used as a starting
        guess in this simple implementation.

    Returns
    -------
    A_opt:
        Optimized MPS tensor with the same shape as ``A_i``.
    """

    chiL, d, chiR = A_i.shape
    H_eff = _build_local_heff(L_i, W_i, R_ip1)

    # Dense Hermitian eigenproblem. For moderate local dimensions this is
    # perfectly adequate and keeps the code compact. If needed, this can
    # be replaced by a Lanczos/Krylov solver acting via a LinearOperator.
    evals, evecs = np.linalg.eigh(H_eff)
    idx_min = np.argmin(evals)
    theta_opt = evecs[:, idx_min]

    A_opt = theta_opt.reshape(chiL, d, chiR)

    # Normalize the updated tensor to avoid unbounded growth of norms.
    nrm = np.linalg.norm(A_opt.ravel())
    if nrm > 0:
        A_opt = A_opt / nrm

    return A_opt


def finite_dmrg(env: Environment, mpo: MPO, mps0: MPS, config: DMRGConfig) -> DMRGResult:
    """Finite-size 1-site DMRG.

    The algorithm keeps the MPS bond dimensions fixed and iteratively
    optimizes one tensor at a time using a 1-site effective Hamiltonian
    built from left/right environments and the MPO.

    A full *sweep* consists of a left-to-right pass followed by a
    right-to-left pass. After each full sweep the energy is evaluated
    using :func:`tensor_network_library.core.utils.expectation_value_env`
    and stored alongside the current bond dimensions.

    Parameters
    ----------
    env:
        Environment describing the system size, local Hilbert space and
        truncation policy. Currently only the size and Hilbert space
        are used; truncation parameters are not needed for 1-site DMRG.
    mpo:
        Hamiltonian as an MPO. Must satisfy ``env.validate_hamiltonian``.
    mps0:
        Initial MPS guess for the ground state. Its bond dimensions
        determine the maximum entanglement that can be represented.
    config:
        DMRGConfig object controlling sweep count and convergence.

    Returns
    -------
    DMRGResult
        Final MPS, list of energies and bond-dimension snapshots per
        sweep.

    Raises
    ------
    ValueError
        If environment, MPO and MPS are incompatible.
    """

    # Basic consistency checks
    env.validate_hamiltonian(mpo)

    if len(mps0) != mpo.L:
        raise ValueError(
            f"MPS length {len(mps0)} does not match MPO length {mpo.L}"
        )
    if mps0.physical_dims != mpo.physical_dims:
        raise ValueError(
            f"MPS physical dims {mps0.physical_dims} do not match MPO dims "
            f"{mpo.physical_dims}"
        )

    # Work on a copy so that the input MPS is not modified unexpectedly.
    mps = mps0.copy()

    # Initial energy estimate using the efficient environment-based helper.
    E_prev = expectation_value_env(mps, mpo)
    energies = [E_prev]
    bond_dims = [mps.bond_dims]

    if config.verbose:
        print(f"[finite_dmrg] initial energy E0 = {E_prev:.12f}")

    L = len(mps)

    for sweep in range(1, config.max_sweeps + 1):
        # --------------------------------------------------------------
        # Left-to-right half-sweep
        # --------------------------------------------------------------
        # Build right environments once from the current MPS. During the
        # left-to-right pass we will update the left environment
        # incrementally.
        R_env = build_right_environments(mps, mpo)
        # Left boundary environment at bond 0
        L_env_i = np.ones((1, 1, 1), dtype=mps.dtype)

        for i in range(L):
            A_i = mps.tensors[i].data
            if A_i is None:
                raise ValueError(f"MPS tensor at site {i} has data=None")
            W_i = mpo.tensors[i].data

            L_i = L_env_i
            R_ip1 = R_env[i + 1]

            A_opt = _optimize_site(L_i, W_i, R_ip1, A_i)
            mps.tensors[i].data = A_opt

            # Update left environment to include the new tensor at site i
            L_env_i = np.einsum(
                "abx,aic,bjd,xijy->cdy",
                L_i,
                A_opt,
                A_opt.conj(),
                W_i,
                optimize=True,
            )

        # --------------------------------------------------------------
        # Right-to-left half-sweep
        # --------------------------------------------------------------
        # Symmetric to the above: build left environments from the updated
        # MPS and update the right environment incrementally as we move
        # from right to left.
        L_env = build_left_environments(mps, mpo)
        R_env_i = np.ones((1, 1, 1), dtype=mps.dtype)

        for offset in range(L):
            i = L - 1 - offset
            A_i = mps.tensors[i].data
            if A_i is None:
                raise ValueError(f"MPS tensor at site {i} has data=None")
            W_i = mpo.tensors[i].data

            L_i = L_env[i]
            R_ip1 = R_env_i

            A_opt = _optimize_site(L_i, W_i, R_ip1, A_i)
            mps.tensors[i].data = A_opt

            # Update right environment to include the new tensor at site i
            R_env_i = np.einsum(
                "cdy,aic,bjd,xijy->abx",
                R_ip1,
                A_opt,
                A_opt.conj(),
                W_i,
                optimize=True,
            )

        # --------------------------------------------------------------
        # End-of-sweep bookkeeping and convergence check
        # --------------------------------------------------------------
        E_curr = expectation_value_env(mps, mpo)
        energies.append(E_curr)
        bond_dims.append(mps.bond_dims)

        if config.verbose:
            dE = E_curr - E_prev
            print(f"[finite_dmrg] sweep {sweep:3d}: E = {E_curr:.12f}, dE = {dE:.3e}")

        if abs(E_curr - E_prev) < config.energy_tol:
            break
        E_prev = E_curr

    return DMRGResult(mps=mps, energies=energies, bond_dims=bond_dims)
