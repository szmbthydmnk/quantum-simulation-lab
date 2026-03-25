"""DMRG example: random transverse X-field Hamiltonian (H2).

H2 = sum_j J_j X_j   with   J_j ~ N(mean=1.0, var=0.1)

Runs
----
    python examples/random_x_field/run_dmrg.py

Physics note
------------
H2 is a sum of single-site X operators.  Its ground state is the
tensor product of |-> = (|0> - |1>)/sqrt(2) at each site (assuming all
J_j > 0).  The exact ground-state energy is -sum_j J_j, which can be
verified analytically.  Like H1, this Hamiltonian is a good DMRG sanity
check: the ground state is a product state and DMRG should converge in
very few sweeps even at chi_max = 1.

Output files (created automatically)
--------------------------------------
    examples/random_x_field/results/H2_convergence.csv
    examples/random_x_field/results/H2_energy_convergence.png
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
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.operators import sigma_x, embed_operator

# ---------------------------------------------------------------------------
# Parameters  — edit freely
# ---------------------------------------------------------------------------
L          = 10
CHI_MAX    = 4       # chi=1 is already exact for H2; use >1 to test larger spaces
MAX_SWEEPS = 20
ENERGY_TOL = 1e-12
MEAN       = 1.0
VAR        = 0.1
SEED       = 7

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_h2(L: int, J: np.ndarray) -> np.ndarray:
    """Build dense H2 = sum_j J_j X_j  (shape 2^L x 2^L)."""
    X = sigma_x()
    H = np.zeros((2**L, 2**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * X, site=j, L=L, d=2)
    return H


def main() -> None:
    rng = np.random.default_rng(SEED)
    J   = rng.normal(loc=MEAN, scale=np.sqrt(VAR), size=L)

    print(f"[H2] L={L}  chi_max={CHI_MAX}")
    print(f"[H2] J = {np.round(J, 4)}")

    # Build MPO with the fixed J values
    X   = sigma_x()
    mpo = MPO.identity_mpo(L=L, d=2, dtype=np.complex128)
    for j in range(L):
        mpo.initialize_single_site_operator(J[j] * X, site=j)

    # Exact reference
    evals, _ = np.linalg.eigh(dense_h2(L, J))
    E_exact  = float(evals[0])
    # Analytic: ground-state energy = -sum_j J_j (all J_j > 0 for small variance)
    E_analytic = -float(np.sum(np.abs(J)))
    print(f"[H2] Exact ground-state energy : {E_exact:.12f}")
    print(f"[H2] Analytic minimum energy   : {E_analytic:.12f}")

    # Initial MPS: |0>^L product state
    # The ground state of H2 is |->^L, which is not |0>^L but is still a
    # product state — DMRG can reach it from |0>^L in O(1) sweeps.
    mps0 = MPS.from_product_state([0] * L, physical_dims=2)
    print(f"[H2] Initial MPS: {mps0}")
    print(f"[H2] Initial energy: "
          f"{expectation_value_env(mps0, mpo):.12f}")

    # Run DMRG
    env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
    config = DMRGConfig(max_sweeps=MAX_SWEEPS, energy_tol=ENERGY_TOL, verbose=True)
    result = finite_dmrg(env, mpo, mps0, config)

    energies  = np.array(result.energies)
    max_bonds = np.array([max(bd) for bd in result.bond_dims])
    sweeps    = np.arange(len(energies))

    E_dmrg  = float(energies[-1])
    E_error = abs(E_dmrg - E_exact)
    print(f"\n[H2] Final DMRG energy   : {E_dmrg:.12f}")
    print(f"[H2] Exact energy        : {E_exact:.12f}")
    print(f"[H2] Analytic min energy : {E_analytic:.12f}")
    print(f"[H2] |E_dmrg - E_exact|  : {E_error:.3e}")

    # Save CSV
    csv_path = OUT_DIR / "H2_convergence.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sweeps, energies.real, max_bonds]),
        delimiter=",",
        header="sweep,energy,max_bond_dim",
        comments="",
    )
    print(f"[H2] Saved CSV  -> {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(sweeps, energies.real, marker="o", label="DMRG")
    ax.axhline(E_exact,    color="red",   ls="--", lw=1.5, label="Exact")
    ax.axhline(E_analytic, color="green", ls=":",  lw=1.5, label="Analytic min")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Energy")
    ax.set_title(r"H2 = $\sum_j J_j X_j$ — energy vs sweep")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax2 = axes[1]
    err = np.clip(np.abs(energies - E_exact), 1e-16, None)
    ax2.semilogy(sweeps, err, marker="s", color="darkorange")
    ax2.set_xlabel("Sweep")
    ax2.set_ylabel(r"$|E_{\rm DMRG} - E_{\rm exact}|$")
    ax2.set_title("Energy error (log scale)")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout()
    plot_path = OUT_DIR / "H2_energy_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H2] Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
