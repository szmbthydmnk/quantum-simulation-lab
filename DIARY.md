# Development Diary

A chronological log of design decisions, bugs found, and milestones reached.

---

## v0.2.1 · 2026-03-26

> Minor follow-up cleanup after the v0.2.0 DMRG milestone.

---

## v0.2.0 · 2026-03-25 — Finite DMRG working end-to-end

**Goal:** Get finite 2-site DMRG converging correctly on all target models.

Started by laying the groundwork for a clean three-layer architecture: `Environment / Hamiltonian / Algorithm`. Wrote `core/site.py` with `QubitSite` and stubs for spin, qutrit, and fermion sites, and `core/geometry.py` with `FiniteChain` and an `InfiniteChain` stub. Refactored `Environment` to own a `geometry` and `system` internally while keeping all existing properties (`L`, `d`, `bc`, `hilbert_dim`, `effective_truncation`) stable so nothing broke downstream. Added `validate_hamiltonian(mpo)` to `Environment` and a thin `Hamiltonian` wrapper with `validate_for(env)`.

The `run_dmrg.py` example for the ZZ+Z model ran and converged cleanly to `E = -9.000000000000` in 9 sweeps. But `pytest` revealed two real bugs:

**Bug 1 — Unicode index label crash**
The einsum string used `a'` (Unicode apostrophe) as an index label in `_update_left_env`. NumPy only accepts plain ASCII letters, throwing `ValueError: Character ' is not a valid symbol` on every call. Fixed by rewriting both environment updaters with unambiguous single-letter indices `{a, b, c, e, s, t, x, y}`.

**Bug 2 — Phantom API in the random X-field MPO**
`_fixed_x_field_mpo` called `MPO.identity_mpo()` and `initialize_single_site_operator()`, neither of which existed anywhere. The MPO it produced was garbage, making every expectation value wrong. Replaced with `random_field_mpo(direction="x")` which already existed and was tested.

Also fixed the deeper sweep-level DMRG bugs from earlier: environments were being rebuilt from scratch inside the bond loop instead of grown incrementally, and the SVD split wasn't enforcing the correct gauge per sweep direction. Both fixed.

**Result:** All 315 tests pass. Cross-validated against iTensor — identical answers.

### Milestones
- ✅ **Phase 1** — `core/site.py`, `core/geometry.py`, refactored `Environment`, `Hamiltonian.validate_for(env)`
- ✅ **Phase 2** — Finite 2-site DMRG with correct incremental environments and gauge, tested on TFIM, Heisenberg, ZZ+Z, random Z-field, random X-field

---

## v0.1.6 · 2026-03-04 — Qubit state library

- Added `tensor_network_library/states/qubit_states` with Pauli eigenstates, Hadamard eigenstates, equator states, and magic state families.
- Added user-facing helpers to list and print available qubit-state labels.
- Integrated into `MPS` via `MPS.from_qubit_labels(...)` for easy product-state initialization.
- Added `MPS.from_state_vector(...)` with successive SVD and `TruncationPolicy` support (cutoff, `max_bond_dim`, strict mode), plus a `from_statevector` compatibility alias.
- Fixed CI LOC injection by adding README markers expected by the `update-loc` workflow.

---

## v0.1.5 · 2026-03-02 — Index tracking

- Developed `Index` class to track tensor contractions properly.
- Updated `Tensor` to accommodate the new indexing method.
- Temporarily disabled non-essential tests outside `test_mps` while rebuilding the contraction layer.

---

## v0.1.3 · 2026-02-21 — SVD

- Added base SVD decomposition method `svd_decomposition` to `tensor.py`.

---

## v0.1.2 · 2026-02-20 — MPS canonicalization

- Extended MPS canonicalization with a left-canonical sweep using QR-based decomposition.
- Added tests checking state preservation and left-orthonormality.
- Prepared infrastructure for SVD-based truncation and right-canonicalization.

---

## v0.1.0 · 2026-02-19 — Foundation

Initial package setup and core primitives.

- Set up `tensor_network_library` as an installable Python package.
- Implemented `Tensor` — numpy wrapper with `copy`, `conj`, `contract`, `reshape`, `transpose`, `norm`, `normalize`, `einsum`, scalar multiplication.
- Implemented `MPS` — site tensors `(χ_L, d, χ_R)`, product-state constructor, norm, normalization, dense statevector conversion.
- Implemented `MPO` — site tensors `(w_L, d_in, d_out, w_R)`, identity constructor, `apply` to MPS, `to_dense` with correct bond contraction and axis ordering.
- Added unit tests: `test_tensor.py`, `test_mps.py`, `test_mpo.py`.
- Configured GitHub Actions CI — `pytest` on all pushes and PRs.
- Created `core_development` branch; merged into `main` via PR #1.
