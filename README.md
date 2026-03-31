# quantum-simulation-lab

[![Tests](https://github.com/szmbthydmnk/quantum-simulation-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/szmbthydmnk/quantum-simulation-lab/actions/workflows/tests.yml)

A small, test-driven tensor network playground focused on **finite-size DMRG** and **MPS/MPO algorithms** for 1D quantum lattice models. Designed for clarity and correctness first, with a strict pytest workflow and cross-checks against iTensor.

---

## Features

- **Core tensor network primitives**
  - `Tensor`, `Index`, `MPS`, `MPO`, `TruncationPolicy`, and utility routines for expectation values and environments.
- **Hamiltonian builders**
  - Transverse-field Ising model (TFIM).
  - XXZ Heisenberg model (via `heisenberg_mpo` and XX wrappers).
  - Random field models in X and Z, ZZ+Z examples.
- **Finite-size 2-site DMRG**
  - Mixed-canonical MPS, left/right environments, and two-site effective Hamiltonians.
  - Incremental environment updates and correct gauge fixing per sweep direction.
  - Tested on TFIM, Heisenberg, random Z-field, random X-field, and ZZ+Z models.
- **Examples and tests**
  - Example scripts under `examples/` with convergence CSVs and plots.
  - 300+ pytest tests, including dense-reference checks and regression tests for DMRG.

Planned (not in v1):

- TEBD and iTEBD (v2).
- Fermionic and higher-spin local Hilbert spaces.
- 2D geometries via 1D mappings and swap gates (v3).

---

## Installation

From source:

```bash
git clone https://github.com/szmbthydmnk/quantum-simulation-lab.git
cd quantum-simulation-lab
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install -e .
```

This installs `quantum-simulation-lab` in editable mode along with its core dependencies (`numpy`, `scipy`, `pytest`, `matplotlib`).[cite:215]

---

## Quickstart: ground state of TFIM with DMRG

```python
import numpy as np

from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.policy import TruncationPolicy
from tensor_network_library.core.utils import expectation_value_env
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.models import tfim_mpo

# System parameters
L = 10
J = 1.0
g = 1.0

# Build environment and Hamiltonian
env = Environment.qubit_chain(L=L, chi_max=32)
mpo = tfim_mpo(L=L, J=J, g=g)

# Initial random MPS
rng = np.random.default_rng(0)
psi = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
psi /= np.linalg.norm(psi)
mps0 = MPS.from_statevector(psi, physical_dims=2, normalize=True)

# DMRG configuration
trunc = TruncationPolicy(max_bond_dim=32)
config = DMRGConfig(max_sweeps=10, energy_tol=1e-10, verbose=True)

result = finite_dmrg(env, mpo, mps0, config)
E0 = result.energies[-1]
print(f"Ground-state energy: {E0:.12f}")
```

See `examples/` for more complete scripts (e.g. random field models and convergence plots).

---

## Running tests and linting

Run the full test suite:

```bash
pytest
```

The GitHub Actions CI runs `pytest` on each push and pull request targeting `main` and keeps the “Lines of code” badge below up to date.[cite:209][cite:210]

---

## Roadmap

High-level plan:

- **v1** – Finite-size DMRG:
  - Robust 2-site DMRG for qubit and spin-1/2 bosonic chains.
  - Canonical forms, incremental environments, dense-reference tests, and iTensor cross-checks.
- **v2** – TEBD / iTEBD:
  - Real and imaginary time evolution on finite and infinite chains.
  - Fermionic and higher-spin site types.
- **v3** – 2D via 1D mappings:
  - Support 2D geometries mapped to 1D chains via swap networks and long-range MPOs.

See [`ROADMAP.md`](./ROADMAP.md) and `DIARY.md` for more detailed development notes.[cite:202]

---

## Lines of code

_Last updated automatically by CI:_

<!-- LOC-AUTO-START -->
5112
<!-- LOC-AUTO-END -->
