# Development Diary

## 2026.02-21 -- v0.1.3

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


