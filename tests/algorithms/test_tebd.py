# tests/algorithms/test_tebd.py

from __future__ import annotations

import numpy as np

from tensor_network_library.algorithms.tebd import (
    TEBDConfig,
    finite_tebd,
    two_site_gate_from_hamiltonian,
)
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.hamiltonian.operators import (
    xx,
    embed_two_site_operator,
)


def _xx_chain_dense(L: int, J: float = 1.0, dtype=np.complex128) -> np.ndarray:
    """
    Dense Hamiltonian for an XX chain:

        H = J Σ_i σ_x^i σ_x^{i+1}

    Built purely from two-site XX couplings.
    """
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)
    h_loc = J * xx(dtype)  # (4, 4) local XX

    for i in range(L - 1):
        H += embed_two_site_operator(h_loc, site=i, L=L, d=d, dtype=dtype)

    return H


def test_apply_single_tebd_step_matches_dense_evolution_small_dt() -> None:
    L = 4
    d = 2
    J = 1.0
    dt = 1e-3  # small dt to make first-order Trotter error negligible

    # Initial random statevector (normalized)
    rng = np.random.default_rng(seed=123)
    psi0 = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
    psi0 = psi0.astype(np.complex128)
    psi0 /= np.linalg.norm(psi0)

    # Exact dense evolution: U_full = exp(-i dt H)
    H_dense = _xx_chain_dense(L=L, J=J)
    evals, evecs = np.linalg.eigh(H_dense)
    phases = np.exp(-1j * dt * evals)
    U_full = (evecs * phases[None, :]) @ evecs.conj().T
    psi_exact = U_full @ psi0

    # Build uniform nearest-neighbour gate from local XX Hamiltonian
    h_two = J * xx(np.complex128)  # (4,4)
    U_two = two_site_gate_from_hamiltonian(h_two, dt=dt)

    # MPS representation of psi0 without truncation
    mps0 = MPS.from_statevector(psi0, physical_dims=d, name="psi0", truncation=None)

    # Truncation policy large enough to keep full rank on this small system
    trunc = TruncationPolicy(max_bond_dim=2**(L // 2))

    # TEBD config: one full step, no explicit renormalization (TEBD is unitary)
    config = TEBDConfig(n_steps=1, normalize=False, verbose=False)

    # For nearest-neighbour uniform chain, even and odd layers use the same gate
    mps_tebd = finite_tebd(
        mps0,
        gates_even=U_two,
        gates_odd=U_two,
        config=config,
        truncation=trunc,
    )

    psi_tebd = mps_tebd.to_dense()

    # Compare up to TEBD + Trotter error; dt is small so tolerance can be tight.
    assert psi_tebd.shape == psi_exact.shape
    assert np.allclose(psi_tebd, psi_exact, atol=1e-5)