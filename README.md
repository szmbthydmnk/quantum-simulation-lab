# quantum-simulation-lab

<div align="center">

[![Tests](https://github.com/szmbthydmnk/quantum-simulation-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/szmbthydmnk/quantum-simulation-lab/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/quantum-simulation-lab?color=0a7c84&label=PyPI)](https://pypi.org/project/quantum-simulation-lab/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green)](./LICENSE)

**A test-driven tensor network library for finite-size DMRG, TEBD, and MPS/MPO algorithms on 1D quantum lattice models.**

[Installation](#installation) · [Quickstart](#quickstart) · [Features](#features) · [Roadmap](./ROADMAP.md) · [Diary](./DIARY.md)

</div>

---

## Overview

`quantum-simulation-lab` is a from-scratch Python implementation of the core tensor network stack needed for **finite-size DMRG** and **TEBD time evolution** on 1D quantum lattice models. The focus is on **correctness and clarity** — every algorithm is covered by a dense test suite and cross-validated against exact diagonalisation and [iTensor](https://itensor.org/).

This project serves as both a learning vehicle and a research platform, built with the same engineering discipline expected in production scientific software: strict CI, typed APIs, and regression tests at every layer.

---

## Features

### 🧱 Core Tensor Network Primitives
- `Tensor`, `Index` — numpy-backed tensors with named indices and full linear algebra support
- `MPS` — matrix product states with product-state, statevector, and qubit-label constructors
- `MPO` — matrix product operators with identity, dense conversion, and MPS application
- `TruncationPolicy` — configurable SVD truncation (cutoff, max bond dimension, strict mode)
- Utility routines for expectation values and left/right environments

### ⚛️ Hamiltonian Builders
- **Transverse-field Ising model** (TFIM)
- **XXZ Heisenberg model** — via `heisenberg_mpo` and XX wrappers
- **Random field models** — X and Z directions, ZZ+Z coupling
- **Transverse Heisenberg model**

### 🔬 Finite-Size 2-Site DMRG
- Mixed-canonical MPS, incremental left/right environments
- Two-site effective Hamiltonians with correct gauge fixing per sweep direction
- Converges on TFIM, Heisenberg, random Z/X fields, and ZZ+Z — results match iTensor

### ⏱️ TEBD Time Evolution _(new in v2)_
- `apply_two_site_gate` — in-place SVD-based two-site gate application with optional truncation
- `two_site_gate_from_hamiltonian` / `two_site_gate_imaginary` — exact-diagonalisation gate builders
- `finite_tebd` — first-order Trotter real-time evolution
- `finite_tebd_strang` — second-order (Strang) Trotter splitting
- `finite_tebd_imaginary` — ground-state preparation via imaginary-time evolution
- `measure_local` — single-site expectation values via transfer matrix sweep

### 🔀 Entangled State Helpers _(new in v2)_
- Bell states (all four), GHZ states, W states — both statevector and MPS form

### 🧪 Test Suite
- **362 pytest tests** — unit, integration, dense-reference checks, DMRG regression tests, and TEBD convergence tests
- CI runs on every push and pull request via GitHub Actions

---

## Installation

```bash
pip install quantum-simulation-lab
```

Or from source:

```bash
git clone https://github.com/szmbthydmnk/quantum-simulation-lab.git
cd quantum-simulation-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quickstart

### Ground State of TFIM with DMRG

```python
import numpy as np
from tensor_network_library.core.env import Environment
from tensor_network_library.core.mps import MPS
from tensor_network_library.algorithms.dmrg import finite_dmrg, DMRGConfig
from tensor_network_library.hamiltonian.models import tfim_mpo

L, J, g = 10, 1.0, 1.0

env = Environment.qubit_chain(L=L, chi_max=32)
mpo = tfim_mpo(L=L, J=J, g=g)

rng = np.random.default_rng(0)
psi = rng.standard_normal(2**L) + 1j * rng.standard_normal(2**L)
mps0 = MPS.from_statevector(psi / np.linalg.norm(psi), physical_dims=2, normalize=True)

config = DMRGConfig(max_sweeps=10, energy_tol=1e-10, verbose=True)
result = finite_dmrg(env, mpo, mps0, config)

print(f"Ground-state energy: {result.energies[-1]:.12f}")
```

### Real-Time Evolution with TEBD

```python
import numpy as np
from tensor_network_library.core.mps import MPS
from tensor_network_library.algorithms.tebd import (
    TEBDConfig,
    two_site_gate_from_hamiltonian,
    finite_tebd,
    measure_local,
)
from tensor_network_library.hamiltonian.models import heisenberg_mpo

L = 10
dt = 0.05
n_steps = 40

# Two-site Heisenberg Hamiltonian (J/4 * (XX + YY + ZZ))
J = 1.0
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H_local = J / 4.0 * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))

# Neel initial state |↑↓↑↓...>
labels = ["up" if i % 2 == 0 else "down" for i in range(L)]
mps0 = MPS.from_qubit_labels(labels)

G = two_site_gate_from_hamiltonian(H_local, dt)
config = TEBDConfig(n_steps=n_steps, normalize=True)
mps_t = finite_tebd(mps0, G, G, config=config)

# Measure <Sz> at each site
Sz = 0.5 * Z
sz_profile = measure_local(mps_t, Sz)
print(f"<Sz> profile at t={n_steps * dt:.2f}: {sz_profile.round(4)}")
```

See [`examples/`](./examples/) for convergence plots and random-field model scripts.

---

## Running Tests

```bash
pytest
```

The CI badge above reflects the current state of the `main` branch.

---

## Roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v1 | Finite-size DMRG — qubit and spin-1/2 chains | ✅ Done |
| v2 | TEBD / iTEBD — real- and imaginary-time evolution | 🔲 In progress |

See [`ROADMAP.md`](./ROADMAP.md) for detailed per-version task lists and [`DIARY.md`](./DIARY.md) for development notes.

---

## Lines of code

_Last updated automatically by CI:_

<!-- LOC-AUTO-START -->
6885
<!-- LOC-AUTO-END -->
