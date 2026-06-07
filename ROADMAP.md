# Roadmap

High-level plan for `quantum-simulation-lab`. Checked items are merged to `main`.

---

## ✅ v1 — Finite-Size DMRG

Robust 2-site DMRG for qubit and spin-1/2 bosonic chains.

- [x] Package structure with `tensor_network_library/core`
- [x] `Tensor`, `Index` — numpy-backed tensors with named indices and full linear algebra
- [x] `MPS` — product-state, statevector, and qubit-label constructors; canonicalization; SVD truncation
- [x] `MPO` — identity, apply-to-MPS, dense conversion
- [x] `TruncationPolicy` — cutoff, max bond dimension, strict mode
- [x] `Environment` — qubit and spin-1/2 bosonic chain support
- [x] `core/site.py` — `QubitSite`
- [x] `core/geometry.py` — `FiniteChain`
- [x] MPO builders — TFIM, Heisenberg (XXZ), random Z/X fields, ZZ+Z
- [x] Finite 2-site DMRG with incremental environments and correct gauge per sweep direction
- [x] Tests — unit, integration, dense-reference, DMRG regression
- [x] Cross-validated against iTensor
- [x] GitHub Actions CI — pytest, LOC badge, PyPI publish

---

## ✅ v2 — TEBD

Real and imaginary-time evolution on finite chains.

- [x] Entangled-state helpers — Bell pair (all 4 states), GHZ, W (dense + MPS wrappers)
- [x] Local two-site gate application on MPS
- [x] `two_site_gate_from_hamiltonian` — exact diagonalisation gate builder
- [x] `two_site_gate_imaginary` — Euclidean gate builder for imaginary-time evolution
- [x] Finite TEBD — first-order Trotter time-stepper
- [x] Finite TEBD — second-order (Strang) Trotter splitting
- [x] Imaginary-time TEBD — ground-state preparation via Euclidean evolution
- [x] `ground_state_search` — high-level ground-state search with convergence tracking
- [x] `measure_local` — single-site expectation values via transfer-matrix sweep
- [x] `measure_bond_energies` — two-site energy expectations via transfer-matrix sandwich
- [x] `CONVENTIONS.md` — tensor and index ordering documentation
- [x] Validated against dense simulations

---
Future extensions beyond the current completed portfolio piece. Not actively developed, but left as potential exploration ideas.

## 🔲 v3+ — TDVP

Potential future possibilities for the project, framed as exploratory ideas rather than current development goals. The repository is considered complete as a portfolio piece.

- [ ] Single-site TDVP (1TDVP) on finite chains
- [ ] Two-site TDVP (2TDVP) with adaptive bond dimension
- [ ] Krylov / Lanczos exponential integrator for the local update step
- [ ] Validated against TEBD for short times and DMRG for ground states

---

## 🔲 v4+ — 2D Geometries & Long-Range Hamiltonians

Optional future extensions for 2D and long-range support, not part of the current portfolio scope.

- [ ] 2D geometries mapped to 1D chains via swap networks
- [ ] Swap gate layer for non-nearest-neighbour couplings
- [ ] Long-range MPO construction (exponential fitting / sum-of-exponentials)
- [ ] Support for heavy-hexagonal lattice geometries
- [ ] Benchmark DMRG ground states on 2D Heisenberg and Hubbard models

