# Roadmap

High-level plan for `quantum-simulation-lab`. Checked items are merged to `main`.

---

## ✅ v0.1 — Core Tensor Network Layer

The foundational data structures and linear algebra layer.

- [x] Package structure with `tensor_network_library/core`
- [x] `Tensor` class with full linear algebra operations
- [x] `MPS` — product-state constructor, norm, dense conversion
- [x] `MPO` — identity, apply-to-MPS, dense conversion
- [x] Unit tests for `Tensor`, `MPS`, and `MPO`
- [x] GitHub Actions CI — `pytest` on every push and PR
- [x] Left-, right-, and mixed-canonicalization routines
- [x] QR decomposition in `Tensor`
- [x] SVD decomposition without truncation
- [x] SVD-based `MPS` constructor from dense statevector with `TruncationPolicy`

---

## ✅ v0.2 — State Helpers, Environments & Finite DMRG

The algorithmic layer: DMRG converging on all target models.

- [x] Qubit state library (`tensor_network_library/states/qubit_states`) — Pauli, Hadamard, equator, magic states
- [x] `MPS.from_qubit_labels(...)` initializer
- [x] `core/site.py` — `QubitSite` and stubs for spin, qutrit, fermion
- [x] `core/geometry.py` — `FiniteChain` and `InfiniteChain` stub
- [x] `Environment` — owns geometry and system; exposes `L`, `d`, `bc`, `hilbert_dim`, `effective_truncation`
- [x] `Hamiltonian.validate_for(env)` wrapper
- [x] MPO builders — TFIM, Heisenberg (XXZ), random Z-field, random X-field, ZZ+Z
- [x] Finite 2-site DMRG with incremental left/right environments and correct gauge per sweep direction
- [x] 315+ tests — unit, integration, dense-reference checks, DMRG regression
- [x] Cross-validated against iTensor
- [ ] Entangled-state helpers (Bell pair, GHZ) for tests
- [ ] Truncation schedule presets (per-sweep bond schedules)

---

## 🔲 v0.3 — TEBD / iTEBD

Time evolution on finite and infinite chains.

- [ ] Local two-site gate application on MPS
- [ ] Simple TEBD time-stepper for nearest-neighbor Hamiltonians
- [ ] Validate TEBD against dense simulations via `to_dense`
- [ ] Tests for unitarity and norm conservation
- [ ] iTEBD on infinite chains

---

## 🔲 v1.0 — Stable Public API

- [ ] TEBD + DMRG + iTEBD behind a clean, versioned public API
- [ ] Full API documentation
- [ ] Performance benchmarks vs. reference implementations

---

## 🔲 v1.5 — Advanced Interfaces (AQA · QML · QEC)

- [ ] Adiabatic quantum algorithm (AQA/AQOA) interfaces using MPS/MPO
- [ ] Parameterized gate hooks for quantum machine learning (QML)
- [ ] QEC encoding/decoding maps as MPOs
- [ ] Tests on small toy problems for all of the above
