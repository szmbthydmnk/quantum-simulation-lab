# tests/algorithms/test_tebd.py
from __future__ import annotations

import numpy as np

from tensor_network_library.algorithms.tebd import (
    TEBDConfig,
    finite_tebd,
    two_site_gate_from_hamiltonian,
    finite_tebd_imaginary,
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
    
# tests/algorithms/test_tebd_imag.py




def _xx_chain_dense(L: int, J: float = 1.0, dtype=np.complex128) -> np.ndarray:
    """
    Dense Hamiltonian for an XX chain:

        H = J Σ_i σ_x^i σ_x^{i+1}

    Built from two-site XX couplings only.
    """
    d = 2
    H = np.zeros((d**L, d**L), dtype=dtype)
    h_loc = J * xx(dtype)  # (4, 4)

    for i in range(L - 1):
        H += embed_two_site_operator(h_loc, site=i, L=L, d=d, dtype=dtype)

    return H


def test_imaginary_time_tebd_converges_to_xx_ground_state_energy() -> None:
    """
    Imaginary-time TEBD on a small XX chain should drive a random
    initial state towards the ground state of H, as seen in the
    expectation value of H.
    """
    L = 6
    d = 2
    J = 1.0
    tau = 0.05          # imaginary-time step
    n_steps = 60        # total β = n_steps * tau

    # Exact dense Hamiltonian and ground-state energy
    H_dense = _xx_chain_dense(L=L, J=J)
    evals, _ = np.linalg.eigh(H_dense)
    E_exact = float(np.min(evals).real)

    # Local two-site Hamiltonian and Euclidean gate
    h_two = J * xx(np.complex128)  # (4, 4)
    U_imag = two_site_gate_from_hamiltonian(h_two, dt=-1j * tau)

    # Random initial MPS (normalized) with a generous chi_max
    chi_max = 2 ** (L // 2)
    mps0 = MPS.from_random(
        L=L,
        chi_max=chi_max,
        physical_dims=d,
        seed=2024,
    )

    trunc = TruncationPolicy(max_bond_dim=chi_max)

    mps_imag = finite_tebd_imaginary(
        mps0=mps0,
        gates_even=U_imag,
        gates_odd=U_imag,
        n_steps=n_steps,
        truncation=trunc,
        verbose=False,
    )

    psi = mps_imag.to_dense()
    psi /= np.linalg.norm(psi)

    E_imag = float(np.vdot(psi, H_dense @ psi).real)

    # Energy should be below or equal to the initial random energy and
    # close to the exact ground-state energy.
    E_init = float(
        np.vdot(mps0.to_dense(), H_dense @ mps0.to_dense()).real
    )
    assert E_imag <= E_init + 1e-8
    assert abs(E_imag - E_exact) < 1e-4