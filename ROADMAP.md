# Roadmap

## v0.1 – Core tensor network layer (DONE)

- [x] Package structure with `tensor_network_library/core`.
- [x] `Tensor` class with basic linear algebra operations.
- [x] `MPS` class with product-state constructor, norms, and dense conversion.
- [x] `MPO` class with identity, apply-to-MPS, and dense conversion.
- [x] Unit tests for Tensor, MPS, and MPO.
- [x] GitHub Actions CI running `pytest` on pushes and PRs.
- [ ] Add left- right- and mied canonicalizations.
- [x] QR decompositionmethod in tensor.
- [ ] SVD decomposition with options for the environment.

## v0.2 – State helpers and environments

- [ ] Add helper functions for common quantum states:
  - [ ] `zero_state(L, d=2)` and `basis_state(L, bits, d=2)` returning MPS.
  - [ ] Simple entangled states for tests (e.g. Bell pair, GHZ).
- [ ] Introduce an environment/config object:
  - [ ] System type (e.g. spin-1/2, spin-1, qudit).
  - [ ] System size `L`, local dimension `d`, boundary conditions.
  - [ ] Bond dimension configuration (static and simple dynamic schedule).
- [ ] Tests validating shapes, norms, and basic config invariants.

## v0.3 – TEBD / iTEBD scaffolding

- [ ] Implement local two-site gate application on MPS.
- [ ] Implement a simple TEBD time-stepper for nearest-neighbor Hamiltonians.
- [ ] Validate TEBD against small dense simulations via `to_dense`.
- [ ] Add tests for unitarity and norm conservation for short evolutions.

## v0.4 – DMRG building blocks

- [ ] Implement effective Hamiltonian construction for a single site / two sites.
- [ ] Integrate a basic eigensolver (SciPy) for local ground-state updates.
- [ ] Add simple 1D model (e.g. transverse-field Ising) as an MPO.
- [ ] Test DMRG ground state energies against exact diagonalization for small systems.
- [ ] Push for a stable v1.0.0 version with TEBD, DMRG, iTEBD

## v1.5 – Advanced features (AQA, QML, QEC, ...)

- [ ] Define interfaces for adiabatic quantum algorithms (AQA/AQOA) using MPS/MPO.
- [ ] Provide simple QML hooks (e.g. parameterized gates that act on MPS).
- [ ] Sketch QEC-related structures (e.g. encoding/decoding maps as MPOs).
- [ ] Extend tests to cover these interfaces on small toy problems.
