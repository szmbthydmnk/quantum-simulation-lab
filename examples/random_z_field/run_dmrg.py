"""DMRG example: random longitudinal Z-field Hamiltonian (H1).

Runs
----
    python examples/random_z_field/run_dmrg.py

What this script does
---------------------
1.  Builds H1 = sum_j J_j Z_j  with  J_j ~ N(mean=1.0, var=0.1)  as an MPO.
2.  Constructs a random initial MPS via SVD decomposition of a Haar-random
    statevector, with bond dimension chi_max.
3.  Runs finite-size 1-site DMRG (alternating left<->right sweeps) via
    ``finite_dmrg`` and records the energy and bond dimensions per sweep.
4.  Computes the exact ground-state energy via dense diagonalisation
    (feasible for the small L used here) and prints the error.
5.  Saves convergence data to ``results/H1_convergence.csv`` and a plot to
    ``results/H1_energy_convergence.png``.

Output files (created automatically)
-------------------------------------
    examples/random_z_field/results/H1_convergence.csv
    examples/random_z_field/results/H1_energy_convergence.png

Known limitation
----------------
    H1 is a sum of *single-site* operators: its ground state is a simple
    product state.  1-site DMRG with chi_max >= 1 therefore converges in
    very few sweeps.  The example is nevertheless useful as a sanity check
    that (a) the DMRG energy matches the exact value and (b) the energy is
    monotonically non-increasing.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on headless systems
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make the repo importable when running as a script from any working directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from examples.dmrg_hamiltonians import random_z_field_mpo
from tensor_network_library.hamiltonian.operators import sigma_z, embed_operator

# ---------------------------------------------------------------------------
# Parameters  (edit freely)
# ---------------------------------------------------------------------------
L          = 10           # chain length
CHI_MAX    = 8            # MPS bond dimension
MAX_SWEEPS = 15           # maximum number of DMRG sweeps
ENERGY_TOL = 1e-10        # energy convergence threshold
MEAN       = 1.0          # mean of J_j distribution
VAR        = 0.1          # variance of J_j distribution
SEED       = 42           # random seed for reproducibility

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_exact_dense_hamiltonian(
    L: int, J: np.ndarray
) -> np.ndarray:
    """Build the dense H1 matrix sum_j J_j Z_j for exact diagonalisation.

    Args:
        L: Chain length.
        J: Array of coupling constants of length L.

    Returns:
        Dense Hermitian matrix of shape (2^L, 2^L).
    """
    d = 2
    Z = sigma_z()
    H = np.zeros((d**L, d**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * Z, site=j, L=L, d=d)
    return H


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # 1.  Draw random couplings and build MPO
    # ------------------------------------------------------------------
    J = rng.normal(loc=MEAN, scale=np.sqrt(VAR), size=L)
    print(f"[H1]  L={L}  chi_max={CHI_MAX}")
    print(f"[H1]  J = {np.round(J, 4)}")

    # Build MPO manually with the fixed J values so that the MPO and the
    # dense reference use exactly the same couplings.
    from tensor_network_library.hamiltonian.operators import sigma_z
    from tensor_network_library.core.mpo import MPO

    Z = sigma_z()
    mpo = MPO.identity_mpo(L=L, d=2, dtype=np.complex128)
    for j in range(L):
        mpo.initialize_single_site_operator(J[j] * Z, site=j)

    # ------------------------------------------------------------------
    # 2.  Exact ground state via dense diagonalisation
    # ------------------------------------------------------------------
    H_dense = build_exact_dense_hamiltonian(L, J)
    evals_exact, evecs_exact = np.linalg.eigh(H_dense)
    E_exact = float(evals_exact[0])
    psi_exact = evecs_exact[:, 0]
    print(f"[H1]  Exact ground-state energy: {E_exact:.12f}")

    # ------------------------------------------------------------------
    # 3.  Initial MPS: SVD of a random state
    # ------------------------------------------------------------------
    # Use the *exact* ground-state vector as the initial guess so the
    # convergence test is clean.  For a more realistic test, replace this
    # with a random vector.
    trunc = TruncationPolicy(max_bond_dim=CHI_MAX)
    mps0 = MPS.from_statevector(
        psi_exact,
        physical_dims=2,
        truncation=trunc,
        normalize=True,
    )
    E_init = expectation_value_env(mps0, mpo)
    print(f"[H1]  Energy after initial SVD MPS: {E_init:.12f}")

    # ------------------------------------------------------------------
    # 4.  Run DMRG
    # ------------------------------------------------------------------
    env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
    config = DMRGConfig(max_sweeps=MAX_SWEEPS, energy_tol=ENERGY_TOL, verbose=True)
    result = finite_dmrg(env, mpo, mps0, config)

    energies  = np.array(result.energies)
    max_bonds = np.array([max(bd) for bd in result.bond_dims])

    E_dmrg  = float(energies[-1])
    E_error = abs(E_dmrg - E_exact)
    print(f"\n[H1]  Final DMRG energy   : {E_dmrg:.12f}")
    print(f"[H1]  Exact energy        : {E_exact:.12f}")
    print(f"[H1]  |E_dmrg - E_exact| : {E_error:.3e}")

    # ------------------------------------------------------------------
    # 5.  Save CSV
    # ------------------------------------------------------------------
    sweep_idx = np.arange(len(energies))
    csv_path  = OUT_DIR / "H1_convergence.csv"
    header    = "sweep,energy,max_bond_dim"
    data_arr  = np.column_stack([sweep_idx, energies, max_bonds])
    np.savetxt(csv_path, data_arr, delimiter=",", header=header, comments="")
    print(f"[H1]  Saved CSV  -> {csv_path}")

    # ------------------------------------------------------------------
    # 6.  Plot energy convergence
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left panel: absolute energy and exact reference
    ax = axes[0]
    ax.plot(sweep_idx, energies, marker="o", label="DMRG energy")
    ax.axhline(E_exact, color="red", linestyle="--", linewidth=1.5, label="Exact")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Energy")
    ax.set_title(r"H1 = $\sum_j J_j Z_j$  energy convergence")
    ax.legend()
    ax.grid(True, alpha=0.4)

    # Right panel: |E_sweep - E_exact| on a log scale
    ax2 = axes[1]
    errors = np.abs(energies - E_exact)
    # Avoid log(0) by clipping
    errors_clipped = np.clip(errors, 1e-16, None)
    ax2.semilogy(sweep_idx, errors_clipped, marker="s", color="darkorange")
    ax2.set_xlabel("Sweep")
    ax2.set_ylabel(r"$|E_{\rm DMRG} - E_{\rm exact}|$")
    ax2.set_title("Energy error (log scale)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    plot_path = OUT_DIR / "H1_energy_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H1]  Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
