"""DMRG algorithms.

This module will host finite-size DMRG implementations built on top of the
core MPS/MPO/Environment infrastructure. The initial version focuses on
well-defined interfaces and reusable utilities; the full two-site sweep
implementation is left as a follow-up step.

The key design goals are:

* Keep all heavy tensor-network logic here, not in tests or examples.
* Reuse environment-based expectation value helpers from
  ``tensor_network_library.core.utils``.
* Expose a small, stable API that tests and examples can depend on.
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

    Attributes:
        max_sweeps:
            Maximum number of full sweeps (left-to-right plus
            right-to-left) to perform.
        energy_tol:
            Convergence tolerance on the change in energy between
            sweeps. When successive energies differ by less than this
            value, the algorithm stops.
        verbose:
            If True, the implementation is allowed to emit diagnostic
            information to stdout (primarily for examples / notebooks).
    """

    max_sweeps: int = 10
    energy_tol: float = 1e-8
    verbose: bool = False


@dataclass
class DMRGResult:
    """Result container for a finite-size DMRG run.

    Attributes:
        mps:
            Final MPS approximation to the ground state.
        energies:
            List of energy estimates recorded after each full sweep.
        bond_dims:
            Snapshot of MPS bond dimensions after each sweep. Each
            entry is the list ``mps.bond_dims`` for that sweep.
    """

    mps: MPS
    energies: List[float]
    bond_dims: List[List[int]]


def finite_dmrg(env: Environment, mpo: MPO, mps0: MPS, config: DMRGConfig) -> DMRGResult:
    """Finite-size DMRG driver (skeleton).

    This function currently provides a *validated* entry point and basic
    bookkeeping (energies and bond dimensions), but does **not yet**
    implement the full two-site sweep algorithm. Instead, it records the
    initial energy and returns without modifying the input MPS.

    The purpose of providing this stub is to allow tests and examples to
    depend on a stable public API while the detailed sweep implementation
    is developed in small, well-tested steps.

    Args:
        env:
            Environment describing the system size, local Hilbert space
            and truncation policy.
        mpo:
            Hamiltonian as an MPO. Must be compatible with ``env``.
        mps0:
            Initial MPS guess for the ground state.
        config:
            DMRGConfig object controlling sweeps and convergence.

    Returns:
        DMRGResult with the (currently unchanged) MPS, a single energy
        entry corresponding to the initial state, and the initial bond
        dimensions.

    Raises:
        ValueError:
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

    # Touch environment builders once to ensure they run without error on
    # the input state. This also guarantees that any shape/convention
    # mismatches are caught early in testing.
    _ = build_left_environments(mps0, mpo)
    _ = build_right_environments(mps0, mpo)

    # Initial energy estimate using the efficient environment-based helper.
    E0 = expectation_value_env(mps0, mpo)

    if config.verbose:
        print(f"[finite_dmrg] initial energy E0 = {E0:.12f}")

    energies = [E0]
    bond_dims = [mps0.bond_dims]

    # TODO: implement left-to-right / right-to-left two-site sweeps here,
    # updating the MPS tensors in-place according to the effective
    # Hamiltonian constructed from left/right environments and the MPO.
    # The final implementation will append new entries to ``energies``
    # and ``bond_dims`` after each full sweep and stop once the energy
    # change falls below ``config.energy_tol``.

    return DMRGResult(mps=mps0, energies=energies, bond_dims=bond_dims)
