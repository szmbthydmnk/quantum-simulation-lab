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
- [x] `core/site.py` — `QubitSite`; stubs for higher spin and fermionic sites
- [x] `core/geometry.py` — `FiniteChain`
- [x] MPO builders — TFIM, Heisenberg (XXZ), random Z/X fields, ZZ+Z
- [x] Finite 2-site DMRG with incremental environments and correct gauge per sweep direction
- [x] 315+ tests — unit, integration, dense-reference, DMRG regression
- [x] Cross-validated against iTensor
- [x] GitHub Actions CI — pytest, LOC badge, PyPI publish

---

## 🔲 v2 — TEBD / iTEBD

Real and imaginary time evolution on finite and infinite chains, with fermionic and higher-spin site support.

- [x] Entangled-state helpers (Bell pair, GHZ)
- [ ] Truncation schedule presets (per-sweep bond schedules)
- [ ] `FermionSite` (spin-1/2) and `SpinSite` (spin-1) implementations
- [ ] Jordan-Wigner string handling for fermionic MPOs
- [ ] Local two-site gate application on MPS
- [ ] Finite TEBD time-stepper for nearest-neighbor Hamiltonians
- [ ] Imaginary-time TEBD for ground-state preparation
- [ ] `InfiniteChain` geometry and iTEBD on infinite chains
- [ ] Validate against dense simulations and known analytical results
- [ ] Tests for unitarity, norm conservation, and fermionic anti-commutation

---

## 🔲 v3 — 2D Geometries & Long-Range Hamiltonians

2D lattice support via 1D mappings, and MPO compression for long-range interactions.

- [ ] 2D geometries mapped to 1D chains via swap networks
- [ ] Swap gate layer for non-nearest-neighbour couplings
- [ ] Long-range MPO construction (exponential fitting / sum-of-exponentials)
- [ ] Support for ladder and cylinder geometries
- [ ] Benchmark DMRG ground states on 2D Heisenberg and Hubbard models

---

## 🔲 v4+ — Advanced Interfaces

Exploratory extensions — scope to be defined.

- [ ] Adiabatic quantum algorithm (AQA) interfaces using MPS/MPO
- [ ] Parameterized gate hooks for quantum machine learning (QML)
- [ ] QEC encoding/decoding maps as MPOs
