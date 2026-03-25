"""DMRG example: Ising-like ZZ+Z Hamiltonian (H3).

H3 = Jz * sum_i Z_i Z_{i+1} - h * sum_i Z_i

We obtain H3 as a special case of the Heisenberg Hamiltonian with
Jx = Jy = 0, only Jz and a longitudinal field h non-zero. The model is
non-trivial (entangling) and provides a more realistic test for DMRG
than purely on-site fields.

Run
---
    python examples/zz_plus_z/run_dmrg.py
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
from tensor_network_library.hamiltonian.models import heisenberg_dense
from examples.dmrg_hamiltonians import zz_plus_z_mpo

# ---------------------------------------------------------------------------
# Parameters  — edit freely
# ---------------------------------------------------------------------------
L          = 10
CHI_MAX    = 32
MAX_SWEEPS = 20
ENERGY_TOL = 1e-10
JZ         = 1.0
H_FIELD    = 1.0
INIT_SEED  = 11

OUT_DIR = pathlib.Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[H3] L={L}  chi_max={CHI_MAX}  Jz={JZ}  h={H_FIELD}")

    mpo = zz_plus_z_mpo(L=L, Jz=JZ, h=H_FIELD)

    # Exact dense reference for small L (here L=20 is borderline but still
    # manageable for demonstration; reduce L for faster runs).
    print("[H3] Building dense reference Hamiltonian (heisenberg_dense)...")
    H_dense = heisenberg_dense(L=L, Jx=0.0, Jy=0.0, Jz=JZ, h=H_FIELD)
    evals, _ = np.linalg.eigh(H_dense)
    E_exact = float(evals[0])
    print(f"[H3] Exact ground-state energy: {E_exact:.12f}")

    # Initial random MPS at bond dimension chi_max
    mps0 = MPS.from_random(L=L, chi_max=CHI_MAX, physical_dims=2, seed=INIT_SEED)
    print(f"[H3] Initial MPS: {mps0}")
    print(f"[H3] Initial energy: {expectation_value_env(mps0, mpo):.12f}")

    env    = Environment.qubit_chain(L=L, chi_max=CHI_MAX)
    config = DMRGConfig(max_sweeps=MAX_SWEEPS, energy_tol=ENERGY_TOL, verbose=True)
    result = finite_dmrg(env, mpo, mps0, config)

    energies  = np.array(result.energies)
    max_bonds = np.array([max(bd) for bd in result.bond_dims])
    sweeps    = np.arange(len(energies))

    E_dmrg  = float(energies[-1])
    E_error = abs(E_dmrg - E_exact)
    print(f"\n[H3] Final DMRG energy   : {E_dmrg:.12f}")
    print(f"[H3] Exact energy        : {E_exact:.12f}")
    print(f"[H3] |E_dmrg - E_exact|  : {E_error:.3e}")

    csv_path = OUT_DIR / "H3_convergence.csv"
    np.savetxt(
        csv_path,
        np.column_stack([sweeps, energies.real, max_bonds]),
        delimiter=",",
        header="sweep,energy,max_bond_dim",
        comments="",
    )
    print(f"[H3] Saved CSV  -> {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(sweeps, energies.real, marker="o", label="DMRG")
    ax.axhline(E_exact, color="red", ls="--", lw=1.5, label="Exact")
    ax.set_xlabel("Sweep")
    ax.set_ylabel("Energy")
    ax.set_title(r"H3 = $J_z \sum Z_i Z_{i+1} - h \sum Z_i$ — energy vs sweep")
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
    plot_path = OUT_DIR / "H3_energy_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[H3] Saved plot -> {plot_path}")


if __name__ == "__main__":
    main()
