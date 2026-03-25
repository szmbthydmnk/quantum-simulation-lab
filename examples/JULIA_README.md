# Julia / ITensors Reference Implementations

These scripts are **ground-truth benchmarks** for the Python DMRG implementation.
They use [ITensors.jl](https://github.com/ITensor/ITensors.jl) and
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), which implement
battle-tested finite-DMRG with 2-site updates and automatic bond-dimension growth.

## Files

| File | Hamiltonian | Python counterpart |
|---|---|---|
| `random_x_field/run_dmrg_itensors.jl` | H2 = Σ J_j X_j | `random_x_field/run_dmrg.py` |
| `random_z_field/run_dmrg_itensors.jl` | H1 = -Σ h_j Z_j | `random_z_field/run_dmrg.py` |

All scripts use the **same seeds** (`SEED=7`, `INIT_SEED=99`), `L=10`, and
`chi_max=4` as the Python examples so energies are directly comparable.

## Setup

```bash
# From the repo root
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This resolves `Project.toml` at the repo root and downloads ITensors,
ITensorMPS, and Plots.

## Running

```bash
# Random X-field (H2)
julia --project=. examples/random_x_field/run_dmrg_itensors.jl

# Random Z-field (H1)
julia --project=. examples/random_z_field/run_dmrg_itensors.jl
```

Results (CSV + PNG) are written to the same `results/` subdirectory as the
Python outputs, with an `_itensors` suffix so they don't clash.

## What to compare

| Quantity | Expected Python | Expected ITensors |
|---|---|---|
| `E_dmrg` | should match | reference value |
| `\|E_dmrg - E_exact\|` | < 1e-10 | < 1e-12 |
| Sweeps to converge | 2–4 | 1–2 |

If Python `|E_dmrg - E_exact|` is large but ITensors is accurate,
the bug is definitively in the Python DMRG engine, not in the Hamiltonian
or the exact-diagonalisation reference.

## Key differences from the Python implementation

- ITensors uses **2-site DMRG** by default, which can grow bond dimensions
  during sweeps. The Python code uses 1-site DMRG (fixed bond dims).
- ITensors represents the Hamiltonian via `OpSum` (AutoMPO), which compresses
  the MPO automatically. The Python code builds the MPO tensor-by-tensor.
- Convergence in ITensors is measured by the energy variance per sweep; the
  Python code uses `|dE|`.
