# tests/algorithms/test_dmrg.py

from __future__ import annotations

import numpy as np

from tensor_network_library.algorithms.dmrg import (
    DMRGConfig,
    finite_dmrg,
)
from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.hamiltonian.models import (
    tfim_mpo,
    tfim_dense,
)


def test_finite_dmrg_tfim_energy_matches_dense_ground_state() -> None:
    """
    Check that finite 2-site DMRG finds the TFIM ground-state energy
    consistent with exact diagonalisation on a small chain.

    Model:
        H = -J Σ_i σ_z^i σ_z^{i+1}  -  g Σ_i σ_x^i
    """
    L = 6
    J = 1.0
    g = 1.3

    # Environment with a modest chi_max; DMRG uses env.effective_truncation.
    env = Environment.qubit_chain(L=L, chi_max=32)

    # MPO Hamiltonian and dense reference
    mpo = tfim_mpo(L=L, J=J, g=g)
    H_dense = tfim_dense(L=L, J=J, g=g)

    evals, _ = np.linalg.eigh(H_dense)
    E_exact = float(np.min(evals).real)

    # Random initial MPS with bond dimension compatible with env
    mps0 = MPS.from_random(
        L=L,
        chi_max=env.max_bond_dim,
        physical_dims=env.d,
        seed=1234,
    )

    config = DMRGConfig(
        max_sweeps=8,
        energy_tol=1e-9,
        verbose=False,
    )

    result = finite_dmrg(
        env=env,
        mpo=mpo,
        mps0=mps0,
        config=config,
        truncation=None,  # use env.effective_truncation
    )

    E_dmrg = result.energies[-1]

    # Ground-state energy should match dense result within a tight tolerance.
    assert np.isclose(E_dmrg, E_exact, rtol=1e-6, atol=1e-6)

    # Bond dimensions should respect the environment cap.
    assert max(result.bond_dims[-1]) <= env.max_bond_dim
    
def test_finite_dmrg_tfim_energy_monotone_along_sweeps() -> None:
    """
    Verify that the DMRG energy does not increase from sweep to sweep
    on a small TFIM chain (within numerical noise).
    """
    L = 6
    J = 1.0
    g = 1.3

    env = Environment.qubit_chain(L=L, chi_max=32)
    mpo = tfim_mpo(L=L, J=J, g=g)

    mps0 = MPS.from_random(
        L=L,
        chi_max=env.max_bond_dim,
        physical_dims=env.d,
        seed=4321,
    )

    config = DMRGConfig(
        max_sweeps=10,
        energy_tol=0.0,   # disable early stopping so we see the full trace
        verbose=False,
    )

    result = finite_dmrg(
        env=env,
        mpo=mpo,
        mps0=mps0,
        config=config,
        truncation=None,
    )

    energies = result.energies  # includes initial energy at index 0
    # Allow for tiny numerical fluctuations but no significant upward jumps.
    for prev, curr in zip(energies, energies[1:]):
        assert curr <= prev + 1e-10