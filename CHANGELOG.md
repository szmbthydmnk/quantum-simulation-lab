# Changelog

All notable changes to `quantum-simulation-lab` are documented here.
This project adheres to [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes that are merged to `main` but not yet tagged._

---

## [2.0.0] — 2026-06-06

### Added
- **`ground_state_search`** (`tebd.py`): high-level imaginary-time TEBD ground-state search.
  Runs a first-order Trotter loop of `exp(-Δτ H)` steps, renormalises after every step, and
  stops when `|E_n − E_{n−1}| < energy_tol`. Returns a `GroundStateResult` with the
  converged MPS, full energy history, norm history, convergence flag, and step count.
- **`GroundStateResult`** dataclass: structured return type for `ground_state_search`.
- **`_left_canonicalize_inplace`** (internal helper): left-to-right QR sweep bringing any
  MPS into left-canonical form in O(L χ² d). Used before every energy measurement in
  `ground_state_search` to ensure Im(⟨H⟩) ≈ 0 regardless of the TEBD gauge.
- **`CONVENTIONS.md`** — tensor and index ordering documentation.
- **Second-order (Strang) Trotter splitting** (`finite_tebd_strang`).

### Fixed
- **Spurious imaginary energy warning in `ground_state_search`**: after a TEBD sweep the MPS
  is in a mixed gauge (S absorbed into the right tensor of the last updated bond). Calling
  `measure_bond_energies` on this state produces non-identity transfer matrices, causing
  Im(⟨H⟩) up to ~0.35 early in the evolution and firing the `> 1e-10` guard ~40 times per
  test run. Fix: `ground_state_search` now QR-sweeps a *copy* of the MPS to left-canonical
  form before every energy measurement; Im(⟨H⟩) drops to machine precision (~1e-14).

### Changed
- `finite_tebd_imaginary` now accepts an optional `measure_fn: (MPS) -> float` callback for
  in-loop energy tracking, populating `TEBDResult.energy_history`.

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
