"""DMRG example: random transverse X-field Hamiltonian (H2).

Runs
----
    python examples/random_x_field/run_dmrg.py

What this script does
---------------------
1.  Builds H2 = sum_j J_j X_j  with  J_j ~ N(mean=1.0, var=0.1)  as an MPO.
2.  Constructs an initial MPS by SVD of the exact ground-state vector,
    truncated to chi_max singular values.
3.  Runs finite-size 1-site DMRG via ``finite_dmrg`` and records energy
    and bond dimensions per sweep.
4.  Computes the exact ground-state energy via dense diagonalisation and
    reports the error.
5.  Saves convergence data to ``results/H2_convergence.csv`` and a plot
    to ``results/H2_energy_convergence.png``.

Output files (created automatically)
--------------------------------------
    examples/random_x_field/results/H2_convergence.csv
    examples/random_x_field/results/H2_energy_convergence.png

Physics note
------------
    H2 is a sum of single-site X operators; its eigenstates are tensor
    products of |+> and |-> states. The ground state has all spins aligned
    in the -X direction.  This Hamiltonian is particularly clean for
    testing because the exact energy is -sum_j J_j (all couplings with the
    same sign as J_j, since the eigenvalue of X is ±1).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.operators import sigma_x, embed_operator

# ---------------------------------------------------------------------------
# Parameters  (edit freely)
# ---------------------------------------------------------------------------
L          = 10
CHI_MAX    = 8
MAX_SWEEPS = 15
ENERGY_TOL = 1e-10
MEAN       = 1.0
VAR        = 0.1
SEED       = 7

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_exact_dense_hamiltonian(
    L: int, J: np.ndarray
) -> np.ndarray:
    """Build the dense H2 matrix sum_j J_j X_j for exact diagonalisation.

    Args:
        L: Chain length.
        J: Array of coupling constants of length L.

    Returns:
        Dense Hermitian matrix of shape (2^L, 2^L).
    """
    d = 2
    X = sigma_x()
    H = np.zeros((d**L, d**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * X, site=j, L=L, d=d)
    return H


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # 1.  Draw random couplings and build MPO
    # ------------------------------------------------------------------
    J = rng.normal(loc=MEAN, scale=np.sqrt(VAR), size=L)
    print(f"[H2]  L={L}  chi_max={CHI_MAX}")
    print(f"[H2]  J = {np.round(J, 4)}")

    X   = sigma_x()
    mpo = MPO.identity_mpo(L=L, d=2, dtype=np.complex128)
    for j in range(L):
        mpo.initialize_single_site_operator(J[j] * X, site=j)

    # ------------------------------------------------------------------
    # 2.  Exact ground state via dense diagonalisation
    # ------------------------------------------------------------------
    H_dense = build_exact_dense_hamiltonian(L, J)
    evals_exact, evecs_exact = np.linalg.eigh(H_dense)
    E_exact  = float(evals_exact[0])
    psi_exact = evecs_exact[:, 0]
    print(f"[H2]  Exact ground-state energy : {E_exact:.12f}")

    # Analytical check: for H2 with all positive J, E_gs = -sum(J)
    E_analytic = -float(np.sum(np.abs(J)))
    print(f"[H2]  Analytic minimum energy   : {E_analytic:.12f}")

    # ------------------------------------------------------------------
    # 3.  Initial MPS: SVD of the exact ground-state vector
    # ------------------------------------------------------------------
    trunc = TruncationPolicy(max_bond_dim=CHI_MAX)
    mps0  = MPS.from_statevector(
        psi_exact,
        physical_dims=2,
        truncation=trunc,
        normalize=True,
    )
    E_init = expectation_value_env(mps0, mpo)
    print(f"[H2]  Energy after initial SVD MPS: {E_init:.12f}")

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
    print(f"\n[H2]  Final DMRG energy   : {E_dmrg:.12f}")
    print(f"[H2]  Exact energy        : {E_exact:.12f}")
    print(f"[H2]  |E_dmrg - E_exact| : {E_error:.3e}")

    # ------------------------------------------------------------------
    # 5.  Save CSV
    # ------------------------------------------------------------------
    sweep_idx = np.arange(len(energies))
    csv_path  = OUT_DIR / "H2_convergence.csv"
    header    = "sweep,energy,max_bond_dim"
    data_arr  = np.column_stack([sweep_idx, energies, max_bonds])
    np.savetxt(csv_path, data_arr, delimiter=",", header=header, comments="")
    print(f"[H2]  Saved CSV  -> {csv_path}")

    # ------------------------------------------------------------------
    # 6.  Plot energy convergence
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(sweep_idx, energies, marker="o", label="DMRG energy")
    ax.axhline(E_exact, color="red", linestyle="--", linewidth=1.5, label="Exact")
    ax.axhline(E_analytic, color="green", linestyle=":", linewidth=1.5, label="Analytic min")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Energy")
    ax.set_title(r"H2 = $\sum_j J_j X_j$  energy convergence")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    errors         = np.abs(energies - E_exact)
    errors_clipped = np.clip(errors, 1e-16, None)
    ax2.semilogy(sweep_idx, errors_clipped, marker="s", color="darkorange")
    ax2.set_xlabel("Sweep")
    ax2.set_ylabel(r"$|E_{\rm DMRG} - E_{\rm exact}|$")
    ax2.set_title("Energy error (log scale)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    plot_path = OUT_DIR / "H2_energy_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H2]  Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
