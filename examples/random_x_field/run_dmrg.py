"""DMRG example: random transverse X-field Hamiltonian (H2).

H2 = Σ_j J_j X_j   with J_j ~ N(mean=MEAN, std=sqrt(VAR))

Physics
-------
H2 is a sum of single-site X operators.  Because every site decouples,
the ground state is the product state |->^L (for all J_j > 0) with
energy E = -sum_j J_j.  The ground state is a product state so chi=1
suffices, but we start at chi_max to test the algorithm at higher bond
dimensions.

MPO construction
----------------
We use ``random_field_mpo`` from ``hamiltonian.models``.  This is a
chi=2 FSM MPO that sums site-local operators correctly.

DO NOT use ``identity_mpo + initialize_single_site_operator`` for this:
that approach *replaces* each identity with the local operator, building
a *product* of operators (tensor product) rather than a *sum*.

Initial MPS
-----------
``MPS.from_random(L, chi_max)`` gives bonds already at chi_max so that
1-site DMRG can optimise the full variational manifold from sweep 1.

Run
---
    python examples/random_x_field/run_dmrg.py
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
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.models import random_field_mpo
from tensor_network_library.hamiltonian.operators import sigma_x, embed_operator

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
L          = 10
CHI_MAX    = 4
MAX_SWEEPS = 20
ENERGY_TOL = 1e-10
MEAN       = 1.0
VAR        = 0.1
SEED       = 7
INIT_SEED  = 99

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dense_h2(L: int, J: np.ndarray) -> np.ndarray:
    """Dense H2 = Σ_j J_j X_j for exact diagonalisation reference."""
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

    # Correct MPO: Σ_j J_j X_j  (chi=2 FSM, not a product of operators)
    mpo = random_field_mpo(L=L, coefficients=J, direction="x")

    # Exact energy via dense diagonalisation
    evals, _   = np.linalg.eigh(dense_h2(L, J))
    E_exact    = float(evals[0])
    E_analytic = -float(np.sum(np.abs(J)))
    print(f"[H2] Exact ground-state energy : {E_exact:.12f}")
    print(f"[H2] Analytic minimum energy   : {E_analytic:.12f}")

    # Random chi_max initial MPS
    mps0 = MPS.from_random(L=L, chi_max=CHI_MAX, physical_dims=2, seed=INIT_SEED)
    print(f"[H2] Initial MPS: {mps0}")
    print(f"[H2] Initial energy: {expectation_value_env(mps0, mpo):.12f}")

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

    csv_path = OUT_DIR / "H2_convergence.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sweeps, energies.real, max_bonds]),
        delimiter=",",
        header="sweep,energy,max_bond_dim",
        comments="",
    )
    print(f"[H2] Saved CSV  -> {csv_path}")

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
    err = np.clip(np.abs(energies.real - E_exact), 1e-16, None)
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
