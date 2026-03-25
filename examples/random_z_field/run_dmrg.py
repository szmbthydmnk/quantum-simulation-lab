"""DMRG example: random longitudinal Z-field Hamiltonian (H1).

H1 = sum_j J_j Z_j   with   J_j ~ N(mean=1.0, var=0.1)

Runs
----
    python examples/random_z_field/run_dmrg.py

Because H1 is a sum of single-site operators its ground state is a
product state (all spins aligned along ±Z depending on sign(J_j)).  It
is exactly representable at bond dimension 1.  The example is useful as a
sanity check: DMRG must converge in very few sweeps and the final energy
must match the exact dense value.

Output files (created automatically)
--------------------------------------
    examples/random_z_field/results/H1_convergence.csv
    examples/random_z_field/results/H1_energy_convergence.png
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
from tensor_network_library.hamiltonian.operators import sigma_z, embed_operator

# ---------------------------------------------------------------------------
# Parameters  — edit freely
# ---------------------------------------------------------------------------
L          = 10      # chain length
CHI_MAX    = 4       # MPS bond dimension (chi=1 is exact for H1, use >1 to test)
MAX_SWEEPS = 20
ENERGY_TOL = 1e-12
MEAN       = 1.0
VAR        = 0.1
SEED       = 42

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_h1(L: int, J: np.ndarray) -> np.ndarray:
    """Build dense H1 = sum_j J_j Z_j  (shape 2^L x 2^L)."""
    Z = sigma_z()
    H = np.zeros((2**L, 2**L), dtype=np.complex128)
    for j in range(L):
        H += embed_operator(J[j] * Z, site=j, L=L, d=2)
    return H


def main() -> None:
    rng = np.random.default_rng(SEED)
    J   = rng.normal(loc=MEAN, scale=np.sqrt(VAR), size=L)

    print(f"[H1] L={L}  chi_max={CHI_MAX}")
    print(f"[H1] J = {np.round(J, 4)}")

    # Build MPO with the fixed J values
    Z   = sigma_z()
    mpo = MPO.identity_mpo(L=L, d=2, dtype=np.complex128)
    for j in range(L):
        mpo.initialize_single_site_operator(J[j] * Z, site=j)

    # Exact reference
    evals, _ = np.linalg.eigh(dense_h1(L, J))
    E_exact  = float(evals[0])
    print(f"[H1] Exact ground-state energy: {E_exact:.12f}")

    # Initial MPS: random product state (all |0>)
    # Using |0>^L as the starting point is fine for H1 because the ground
    # state is a product state; for more entangled Hamiltonians you should
    # use a random bond-chi MPS.
    mps0 = MPS.from_product_state([0] * L, physical_dims=2)
    print(f"[H1] Initial MPS: {mps0}")
    print(f"[H1] Initial energy (from_product_state): "
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
    print(f"\n[H1] Final DMRG energy   : {E_dmrg:.12f}")
    print(f"[H1] Exact energy        : {E_exact:.12f}")
    print(f"[H1] |E_dmrg - E_exact|  : {E_error:.3e}")

    # Save CSV
    csv_path = OUT_DIR / "H1_convergence.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sweeps, energies, max_bonds]),
        delimiter=",",
        header="sweep,energy,max_bond_dim",
        comments="",
    )
    print(f"[H1] Saved CSV  -> {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(sweeps, energies.real, marker="o", label="DMRG")
    ax.axhline(E_exact, color="red", ls="--", lw=1.5, label="Exact")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Energy")
    ax.set_title(r"H1 = $\sum_j J_j Z_j$ — energy vs sweep")
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
    plot_path = OUT_DIR / "H1_energy_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H1] Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
