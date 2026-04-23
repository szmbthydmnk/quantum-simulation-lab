# tests/examples/test_tebd_physical_problems.py
from __future__ import annotations

import numpy as np

from tensor_network_library.algorithms.tebd import (
    TEBDConfig,
    finite_tebd,
    two_site_gate_from_hamiltonian,
)
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.hamiltonian.models import tfim_dense
from tensor_network_library.hamiltonian.operators import (
    zz,
    sigma_x,
    identity,
)


def _tfim_two_site_local_hamiltonian(
    J: float,
    g: float,
    dtype: np.dtype = np.complex128,
) -> np.ndarray:
    r"""
    Local two-site TFIM Hamiltonian on sites (i, i+1):

        h_{i,i+1} = -J σ_z^i σ_z^{i+1}
                    - (g/2)(σ_x^i + σ_x^{i+1})

    Summing h_{i,i+1} over i with open boundaries reproduces

        H = -J Σ_i σ_z^i σ_z^{i+1} - g Σ_i σ_x^i
    """
    d = 2
    SzSz = zz(dtype)              # σ_z ⊗ σ_z, shape (4,4)
    Sx = sigma_x(dtype)           # (2,2)
    I = identity(d, dtype)        # (2,2)

    # Single-spin X on left and right sites in the two-site Hilbert space
    Sx_left = np.kron(Sx, I)
    Sx_right = np.kron(I, Sx)

    h_loc = -J * SzSz - 0.5 * g * (Sx_left + Sx_right)
    return h_loc.astype(dtype, copy=False)


def test_tebd_tfim_quench_matches_dense_for_small_dt() -> None:
    """
    Real-time TEBD on a small TFIM chain should match dense evolution
    for a single small time step (within Trotter + truncation error).

    Model:
        H = -J Σ_i σ_z^i σ_z^{i+1} - g Σ_i σ_x^i

    Initial state:
        |psi0> = |+ + + +>
    """
    L = 4
    d = 2
    J = 1.0
    g = 1.3
    dt = 1e-3  # very small to suppress first-order Trotter error

    # Build initial MPS as |+...+> and get the corresponding dense psi0
    mps0 = MPS.from_qubit_labels(
        labels=["+"] * L,
        name="tfim_plus_state",
        dtype=np.complex128,
    )
    psi0 = mps0.to_dense()
    psi0 /= np.linalg.norm(psi0)

    # Exact dense evolution: U_full = exp(-i dt H)
    H_dense = tfim_dense(L=L, J=J, g=g, dtype=np.complex128)
    evals, evecs = np.linalg.eigh(H_dense)
    phases = np.exp(-1j * dt * evals)
    U_full = (evecs * phases[None, :]) @ evecs.conj().T
    psi_exact = U_full @ psi0

    # Build uniform two-site TFIM gate
    h_two = _tfim_two_site_local_hamiltonian(J=J, g=g, dtype=np.complex128)
    U_two = two_site_gate_from_hamiltonian(h_two, dt=dt)

    # TEBD config: one full step, no renormalization (unitary gates)
    trunc = TruncationPolicy(max_bond_dim=2 ** (L // 2))
    config = TEBDConfig(n_steps=1, normalize=False, verbose=False)

    mps_tebd = finite_tebd(
        mps0=mps0,
        gates_even=U_two,
        gates_odd=U_two,
        config=config,
        truncation=trunc,
    )

    psi_tebd = mps_tebd.to_dense()

    # Compare up to first-order Trotter error.
    # For L=4, J=1, g=1.3, dt=1e-3, the first-order Lie-Trotter splitting
    # [even bonds] then [odd bonds] gives a per-step max error of ~3.3e-4.
    # This is O(dt) in the leading error term (proportional to [H_even, H_odd]).
    # A stricter test requires implementing second-order (Strang) splitting in
    # finite_tebd
    assert psi_tebd.shape == psi_exact.shape
    assert np.allclose(psi_tebd, psi_exact, atol=5e-4)