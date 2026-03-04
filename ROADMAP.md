# Roadmap

## v0.1 – Core tensor network layer (DONE)

- [x] Package structure with `tensor_network_library/core`.
- [x] `Tensor` class with basic linear algebra operations.
- [x] `MPS` class with product-state constructor, norms, and dense conversion.
- [x] `MPO` class with identity, apply-to-MPS, and dense conversion.
- [x] Unit tests for Tensor, MPS, and MPO.
- [x] GitHub Actions CI running `pytest` on pushes and PRs.
- [ ] Add left-, right-, and mixed-canonicalization routines.
- [x] QR decomposition method in `Tensor`.
- [x] SVD decomposition without truncation.
- [x] SVD-based MPS constructor from dense statevector with `TruncationPolicy` (cutoff + `max_bond_dim`).

## v0.2 – State helpers and environments

- [x] Add helper functions for common qubit states (`tensor_network_library/states/qubit_states`).
- [x] Add an MPS initializer from qubit labels (`MPS.from_qubit_labels`).
- [ ] Add simple entangled states for tests (e.g. Bell pair, GHZ).
- [ ] Introduce an environment/config object:
  - [ ] System type (e.g. spin-1/2, spin-1, qudit).
  - [ ] System size `L`, local dimension `d`, boundary conditions.
  - [ ] Truncation policy presets/schedules (e.g. per-sweep bond schedules).
- [ ] Tests validating shapes, norms, and basic config invariants.

## v0.3 – TEBD / iTEBD scaffolding

- [ ] Implement local two-site gate application on MPS.
- [ ] Implement a simple TEBD time-stepper for nearest-neighbor Hamiltonians.
- [ ] Validate TEBD against small dense simulations via `to_dense`.
- [ ] Add tests for unitarity and norm conservation for short evolutions.

## v0.4 – DMRG building blocks

- [ ] Operator helpers similar to `qubit_states` (single-site ops like X/Y/Z; two-site couplings; utilities to assemble Hamiltonians).
- [ ] MPO builders for standard 1D models (e.g. transverse-field Ising, Heisenberg).
- [ ] Canonical forms + environment tensors (left/right blocks) for sweeps.
- [ ] Implement effective Hamiltonian construction for a single site / two sites.
- [ ] Integrate a basic eigensolver (SciPy) for local ground-state updates.
- [ ] Test DMRG ground-state energies against exact diagonalization for small systems.

## v1.0 – Stable algorithms

- [ ] TEBD + DMRG + iTEBD in a stable public API.

## v1.5 – Advanced features (AQA, QML, QEC, ...)

- [ ] Define interfaces for adiabatic quantum algorithms (AQA/AQOA) using MPS/MPO.
- [ ] Provide simple QML hooks (e.g. parameterized gates that act on MPS).
- [ ] Sketch QEC-related structures (e.g. encoding/decoding maps as MPOs).
- [ ] Extend tests to cover these interfaces on small toy problems.
