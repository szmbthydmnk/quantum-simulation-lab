"""DMRG example: random longitudinal Z-field Hamiltonian (H1).

H1 = sum_j J_j Z_j   with J_j ~ N(mean=MEAN, std=sqrt(VAR))

The ground state of H1 is a product state |s0 s1 ... s_{L-1}> where
s_j = 0 if J_j > 0, s_j = 1 if J_j < 0.  It is exactly representable
at bond dimension 1.

Initial guess
-------------
We start from a *random* MPS at chi_max to avoid symmetry-protected
zero modes.  DMRG should converge to machine precision in O(1) sweeps.

Runs
----
    python examples/random_z_field/run_dmrg.py
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
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import DMRGConfig, finite_dmrg
from tensor_network_library.hamiltonian.models import random_field_mpo
from tensor_network_library.hamiltonian.operators import sigma_z, embed_operator

# ---------------------------------------------------------------------------
# Parameters  — edit freely
# ---------------------------------------------------------------------------
L          = 10
CHI_MAX    = 4       # chi=1 is exact for H1; chi=4 lets us test larger spaces
MAX_SWEEPS = 20
ENERGY_TOL = 1e-12
MEAN       = 1.0
VAR        = 0.1
SEED       = 42
INIT_SEED  = 0       # separate seed for the initial random MPS

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

    # Correct MPO: Σ_j J_j Z_j as a sum, not a product.
    mpo = random_field_mpo(L=L, coefficients=J, direction="z")

    # Exact reference
    evals, _ = np.linalg.eigh(dense_h1(L, J))
    E_exact  = float(evals[0])
    print(f"[H1] Exact ground-state energy: {E_exact:.12f}")

    # Initial MPS: random bond-chi_max state to avoid zero-energy subspaces.
    mps0 = MPS.from_random(L=L, chi_max=CHI_MAX, physical_dims=2, seed=INIT_SEED)
    print(f"[H1] Initial MPS: {mps0}")
    print(f"[H1] Initial energy: {expectation_value_env(mps0, mpo):.12f}")

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
        np.column_stack([sweeps, energies.real, max_bonds]),
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
