# Development Diary

## 2026-03-04 -- v0.1.6
- Added qubit-state constructors and parsers in `tensor_network_library/states/qubit_states` (including Pauli eigenstates, Hadamard eigenstates, equator states, and magic state families).
- Added user-facing helpers to list and print available qubit-state labels.
- Integrated qubit states into `MPS` via `MPS.from_qubit_labels(...)` for easy product-state initialization.
- Added `MPS.from_state_vector(...)` constructor using successive SVD with `TruncationPolicy` support (cutoff, `max_bond_dim`, and `strict` behavior), plus a compatibility alias `from_statevector`.
- Added/extended pytest coverage for the new state helpers and MPS constructors.
- Fixed CI LOC injection by adding README markers expected by the `update-loc` workflow.

## 2026-03-02 -- 2026-03-03 -- v0.1.5
- Developed `index` in order to track contractions properly.
- Updated `tensor` to accomodate the new indexing method.
- Working on `mps`, therefore I disabled all non-necessary tests outside the `test_mps`

## 2026-02-

## 2026-02-21 -- v0.1.3

- Added a base SVD decomposition method to tensor.py (`svd_decomposition`)

## 2026-02-20 -- v0.1.2
- Extended MPS canonicalization layer with left-canonical sweep using QR-based decomposition.
- Added canonicalization tests checking state preservation and left-orthonormality.
- Prepared infrastructure for future SVD-based truncation and right-canonicalization (design and API adjustments).
- Bumped core version to `v0.1.2` for this development snapshot.
  
## 2026-02-19 / 2026-02-20

- Set up `tensor_network_library` as a Python package with `pyproject.toml` and editable install.
- Implemented `Tensor` core class in `tensor_network_library/core/tensor.py` with:
  - Basic wrapper around `numpy.ndarray` and complex dtype.
  - Methods: `copy`, `conj`, `contract`, `reshape`, `transpose`, `norm`, `normalize`, `einsum`, scalar multiplication.
- Implemented `MPS` class in `tensor_network_library/core/mps.py`:
  - Representation as a list of site tensors `(chi_left, d, chi_right)`.
  - Product-state constructor `from_product_state`.
  - Norm and in-place normalization.
  - Conversion to dense statevector `to_dense`.
- Implemented `MPO` class in `tensor_network_library/core/mpo.py`:
  - Representation as a list of site tensors `(wL, d_in, d_out, wR)`.
  - Identity constructor `identity`.
  - Application to MPS `apply`.
  - Conversion to dense operator `to_dense` with explicit bond contraction and axis ordering.
- Added unit tests under `tests/`:
  - `test_tensor.py` for Tensor operations.
  - `test_mps.py` for MPS product states and norms.
  - `test_mpo.py` for MPO identity, dense conversion, and apply consistency.
- Configured GitHub Actions CI (`.github/workflows/tests.yml`) to run `pytest` on pushes and PRs.
- Created `core_development` branch and merged the initial core TN layer into `main` via PR #1.

