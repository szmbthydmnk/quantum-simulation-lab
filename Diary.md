# Development Diary

---

## 25 March 2026

Today was a productive session. The main goal was to get finite DMRG working properly — and we actually got there.

Started by laying the groundwork for the three-layer architecture (Environment / Hamiltonian / Algorithm). Wrote out the design for `core/site.py` with the `QubitSite` and stubs for spin, qutrit, and fermion sites, and `core/geometry.py` with `FiniteChain` and an `InfiniteChain` stub. Refactored `Environment` to own a `geometry` and `system` internally while keeping all the existing properties (`L`, `d`, `bc`, `hilbert_dim`, `effective_truncation`) stable so nothing broke downstream. Added `validate_hamiltonian(mpo)` to `Environment` and a thin `Hamiltonian` wrapper with `validate_for(env)` on top of the existing MPO builders.

Then moved on to DMRG itself. The `run_dmrg.py` example for the ZZ+Z model ran and converged cleanly to E = -9.000000000000 in 9 sweeps, which was a good sign. But `pytest` revealed two real bugs:

**First bug** was a crash in `_update_left_env` — the einsum string used `a'` as an index label (Unicode apostrophe). NumPy only accepts plain ASCII letters, so it threw a `ValueError: Character ' is not a valid symbol` on every single call. Fixed by rewriting both environment updaters with unambiguous single-letter indices `{a, b, c, e, s, t, x, y}`.

**Second bug** was in the H2 (random X-field) test — `_fixed_x_field_mpo` was calling `MPO.identity_mpo()` and `initialize_single_site_operator()`, neither of which exist anywhere in the codebase. It was a phantom API, so the MPO it produced was garbage and the expectation value was completely wrong. Replaced it with `random_field_mpo(direction="x")` which already existed and was already tested.

The deeper DMRG bugs from earlier (oscillating energy between sweeps) were also addressed today: environments were previously being rebuilt from scratch inside the bond loop instead of grown incrementally, and the SVD split wasn't enforcing the correct gauge depending on sweep direction. Both fixed.

All 315 tests pass. Results checked against iTensor — same answers.

Opened PR #25 (`core_development` → `main`) with all of this.

### Milestones ticked off today

- **Phase 1 complete** — `core/site.py`, `core/geometry.py`, refactored `Environment`, `Hamiltonian.validate_for(env)`, tests for all of the above
- **Phase 2 complete** — finite 2-site DMRG with correct incremental environments and gauge, tested on TFIM, Heisenberg, ZZ+Z, random Z-field, random X-field, validated against iTensor

### Up next

- Phase 3: TEBD — Suzuki-Trotter gates, SVD truncation after each gate, nearest-neighbour only to start
