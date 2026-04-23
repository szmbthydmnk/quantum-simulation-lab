# Changelog

All notable changes to `quantum-simulation-lab` are documented here.
This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes that are merged to `main` but not yet tagged._

- iTEBD on infinite chains (`InfiniteChain` geometry + `itebd.py`)
- Second-order (Strang) Trotter splitting for finite TEBD
- Truncation schedule presets
- Benchmark suite: ED vs DMRG, TEBD vs ED, DMRG/TEBD vs iTensor, iTEBD validation
- `CONVENTIONS.md` — tensor and index ordering documentation

---

## [1.2.2] — 2026-04-23

### Fixed
- CI `publish.yml` heredoc scoping bug: `NEW_VERSION` shell variable was out of scope across `run:` steps; replaced with Actions expression syntax `${{ steps.version.outputs.version }}` substituted by the runner before shell execution.

---

## [1.2.1] — 2026-04-23

### Fixed
- `ComplexWarning` in `utils.py` and `test_dmrg_hamiltonians.py` caused by implicit cast from complex to float in `np.vdot` calls; cast is now explicit.
- Flaky `test_strang_more_accurate_than_first_order` test: tightened dt and tolerance so second-order convergence is reliably distinguishable from first-order.

---

## [1.2.0] — 2026-04-23

### Added
- **Entangled-state helpers** (`tensor_network_library/states/entangled_states.py`):
  - All four Bell states as statevectors and as MPS
  - GHZ states for arbitrary `L` as statevectors and MPS
  - W states for arbitrary `L` as statevectors and MPS
  - Public re-exports via `tensor_network_library/__init__.py`
- **Two-site gate application** (`apply_two_site_gate`): in-place SVD-based gate on adjacent MPS sites with optional `TruncationPolicy`.
- **Gate builders**:
  - `two_site_gate_from_hamiltonian(H, dt)` — real-time gate via exact diagonalisation of a 4×4 local Hamiltonian.
  - `two_site_gate_imaginary(H, dt)` — imaginary-time gate (non-unitary) for ground-state preparation.
- **Finite TEBD** (`finite_tebd`): first-order Trotter time-stepper sweeping even/odd bond layers.
- **Imaginary-time TEBD** (`finite_tebd_imaginary`): Euclidean time evolution converging to the ground state; validated against DMRG energies for TFIM and Heisenberg.
- **`measure_local`**: single-site expectation values via efficient transfer-matrix sweep without forming the full statevector.
- **`TEBDConfig`**: dataclass for step count, truncation policy, and normalisation flag.
- **Transverse Heisenberg MPO builder** (`transverse_heisenberg_mpo`).

### Changed
- `MPS` constructors unified: `from_statevector`, `from_qubit_labels`, and product-state paths now all pass through a single canonicalisation routine.
- `TruncationPolicy` gains a `strict` flag; truncation errors are now returned from `apply_two_site_gate` for downstream inspection.

### Tests
- 47 new tests covering entangled states, gate builders, TEBD convergence, and imaginary-time evolution.
- Total test count: **362**.

---

## [1.0.4] — 2026-03-08

### Fixed
- Edge-case in right-to-left DMRG sweep: gauge was not restored after last site update, causing incorrect energies on subsequent sweeps for open boundary conditions with `L=4`.
- `MPO.to_dense()` memory layout was transposing physical indices for `L > 8`; now consistent with statevector qubit ordering.

---

## [1.0.3] — 2026-03-08

### Added
- `MPO.apply(mps)` — applies an MPO to an MPS and returns a new MPS, used as the basis for the DMRG effective Hamiltonian contraction.
- ZZ+Z random field Hamiltonian builder.
- Random X-field Hamiltonian builder.

### Fixed
- `Environment.update_left` / `update_right` were not normalising the boundary tensors, leading to numerical drift over many sweeps.

---

## [1.0.2] — 2026-03-08

### Added
- `heisenberg_mpo` and `xx_mpo` Hamiltonian builders for XXZ and isotropic XX models.
- Finite 2-site DMRG converges on Heisenberg and random-field models; energies match iTensor to `1e-8`.

---

## [1.0.0] — 2026-03-04

### Added
- Initial release.
- `Tensor`, `Index` — numpy-backed tensors with named indices.
- `MPS` — product-state, statevector, and qubit-label constructors; left/right canonicalisation; SVD truncation via `TruncationPolicy`.
- `MPO` — identity construction and dense conversion.
- `Environment` — qubit chain and spin-1/2 bosonic chain; incremental left/right environment updates.
- `FiniteChain` geometry, `QubitSite` site type.
- `tfim_mpo` — transverse-field Ising model MPO.
- Finite 2-site DMRG (`finite_dmrg`) with `DMRGConfig`; converges on TFIM; energies match iTensor.
- GitHub Actions CI: pytest, LOC auto-badge, PyPI trusted publishing.
- 315 tests at initial release.
