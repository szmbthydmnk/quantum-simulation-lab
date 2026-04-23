# Conventions

> **Scope:** This document records every tensor-ordering, index-naming, sign, and algorithmic convention used in `quantum-simulation-lab`. It is the single source of truth. When code and this document disagree, fix the code.

---

## Table of Contents

1. [Site and chain labelling](#1-site-and-chain-labelling)
2. [Tensor axis ordering](#2-tensor-axis-ordering)
3. [Index naming](#3-index-naming)
4. [Bond dimensions](#4-bond-dimensions)
5. [Canonicalisation gauge](#5-canonicalisation-gauge)
6. [SVD and singular-value absorption](#6-svd-and-singular-value-absorption)
7. [Gate ordering](#7-gate-ordering)
8. [MPO structure](#8-mpo-structure)
9. [Hamiltonian sign conventions](#9-hamiltonian-sign-conventions)
10. [Statevector qubit ordering](#10-statevector-qubit-ordering)
11. [DMRG conventions](#11-dmrg-conventions)
12. [TEBD conventions](#12-tebd-conventions)
13. [Imaginary-time evolution](#13-imaginary-time-evolution)
14. [Measurement conventions](#14-measurement-conventions)
15. [Dtype and precision](#15-dtype-and-precision)

---

## 1. Site and chain labelling

- Sites are labelled $0, 1, \ldots, L-1$ (zero-indexed, left to right).
- **Open boundary conditions (OBC)** are the default for all finite algorithms. Periodic boundaries are not yet implemented.
- Bond $b$ connects site $b-1$ to site $b$. Bond $0$ is the left virtual vacuum (dimension 1); bond $L$ is the right virtual vacuum (dimension 1).
- A chain of length $L$ has $L$ site tensors and $L+1$ bonds (including both boundary bonds of dimension 1).

```
bond_0  site_0  bond_1  site_1  bond_2  ...  bond_{L-1}  site_{L-1}  bond_L
  [1] ── [A_0] ── [χ_1] ── [A_1] ── [χ_2] ── ... ── [χ_{L-1}] ── [A_{L-1}] ── [1]
```

---

## 2. Tensor axis ordering

### MPS site tensor

Every site tensor $A_i$ is a rank-3 array with axes in this fixed order:

```
A_i.data.shape  ==  (χ_left,  d_i,  χ_right)
                      axis 0   axis 1   axis 2
```

| Axis | Role | Symbol |
|------|------|--------|
| 0 | Left virtual bond | $\chi_\text{left}$ |
| 1 | Physical (local Hilbert space) | $d_i$ |
| 2 | Right virtual bond | $\chi_\text{right}$ |

This matches the `_create_empty_tensors` ordering in `mps.py`:
```python
inds = [self.bonds[i], self.indices[i], self.bonds[i + 1]]
```

### MPO site tensor

Every MPO site tensor $W_i$ is rank-4:

```
W_i.shape  ==  (χ_left,  d_i,  d_i,  χ_right)
                axis 0   axis 1  axis 2  axis 3
```

| Axis | Role |
|------|------|
| 0 | Left MPO virtual bond |
| 1 | Physical ket (output) index |
| 2 | Physical bra (input) index |
| 3 | Right MPO virtual bond |

The ket index (axis 1) always comes before the bra index (axis 2).

### Two-site (theta) tensor

When two adjacent MPS tensors are contracted together during a gate application or DMRG update:

```
theta[a, i, j, b]  ==  sum_c  A_i[a, i, c] * A_j[c, j, b]
```

Shape: `(χ_left, d_i, d_j, χ_right)` — left bond, left physical, right physical, right bond.

This is the ordering used in `gate_application.py`:
```python
theta = np.einsum("aic,cjb->aijb", A_i, A_j)
```

### Gate tensor (rank-4)

A two-site gate acting on sites $(i, i+1)$ is stored as:

```
U[i', j', i, j]
```

where primed indices are **output (ket)** and unprimed are **input (bra)**. As a matrix it is flattened as `U.reshape(d*d, d*d)` with row index $(i', j')$ and column index $(i, j)$ — standard operator convention.

```python
# gate_application.py applies it as:
theta_prime = np.einsum("mnij,aijb->amnb", U, theta)
#               U[m=i', n=j', i, j] * theta[a, i, j, b]
```

---

## 3. Index naming

`Index` objects carry a human-readable `name` and a frozen set of `tags`. The naming scheme:

| Index type | Name pattern | Tags |
|-----------|--------------|------|
| Physical index at site $i$ | `{mps_name}_phys_{i}` | `{"phys", "i={i}"}` |
| Bond between sites $i$ and $i+1$ | `{mps_name}_bond_{i+1}` | `{"bond", "b={i+1}"}` |
| Left boundary bond | `{mps_name}_bond_0` | `{"bond", "b=0"}` |
| Right boundary bond | `{mps_name}_bond_{L}` | `{"bond", "b={L}"}` |

Bond index $b$ sits between site $b-1$ and site $b$, so `bond_1` is shared between `tensors[0]` (as its `axis 2`) and `tensors[1]` (as its `axis 0`).

After a gate application, the shared bond between site $i$ and $i+1$ is recreated with the same name and tags as the original `bonds[i+1]` but with the updated dimension `chi_new`.

---

## 4. Bond dimensions

- `mps.bonds` is a list of `Index` objects of length $L+1$. `bonds[b].dim` gives $\chi_b$.
- `mps._bond_dims` mirrors this as a plain `List[int]` for fast arithmetic.
- Boundary bonds: `bonds[0].dim == bonds[L].dim == 1` always.
- **Default bond policy** (`bond_policy="default"`): $\chi_b = \min\!\left(\prod_{k<b} d_k,\; \prod_{k \geq b} d_k\right)$, optionally capped by `TruncationPolicy.max_bond_dim`.
- **Uniform bond policy** (`bond_policy="uniform"`): all interior bonds set to `truncation.max_bond_dim`; boundary bonds remain 1.

### TruncationPolicy

`TruncationPolicy` (in `core/policy.py`) controls SVD truncation in both `from_statevector` and `apply_two_site_gate`:

- `max_bond_dim`: hard cap on $\chi$ after SVD.
- `svd_cutoff`: singular values $\sigma < \text{cutoff}$ are discarded (default `1e-12`).
- `strict`: if `True`, raises on truncation error above threshold (default `False`).

At least one singular value is always kept, even if all fall below the cutoff.

---

## 5. Canonicalisation gauge

### Left-canonical tensor

$A_i$ is **left-canonical** if:

$$\sum_{a,\sigma} (A_i)^*_{a\sigma c}\,(A_i)_{a\sigma c'} = \delta_{cc'}$$

i.e. `A_i.reshape(χ_left * d, χ_right)` has orthonormal columns.

### Right-canonical tensor

$B_i$ is **right-canonical** if:

$$\sum_{\sigma,c} (B_i)^*_{a\sigma c}\,(B_i)_{a'\sigma c} = \delta_{aa'}$$

i.e. `B_i.reshape(χ_left, d * χ_right)` has orthonormal rows.

### Mixed-canonical (site-canonical) form

An MPS is in **mixed-canonical form centred at site $k$** when:
- Sites $0, \ldots, k-1$ are left-canonical.
- Sites $k+1, \ldots, L-1$ are right-canonical.
- Site $k$ is unconstrained (carries all norm).

This is the standard gauge used by the DMRG local update: the effective Hamiltonian at site $k$ is orthogonally projected by the left/right environments.

### Canonicalisation in `canonical.py`

`left_canonicalize(mps, site)` and `right_canonicalize(mps, site)` sweep QR decompositions across the chain. Singular values from QR are absorbed **rightward** during left-sweep, **leftward** during right-sweep. The norm is concentrated at the orthogonality centre.

---

## 6. SVD and singular-value absorption

When splitting a two-site tensor $\Theta$ via SVD:

$$\Theta = U\, S\, V^\dagger$$

the singular values $S$ must be absorbed into one of the two new tensors. The `absorb` parameter controls this:

| `absorb` value | Left tensor $A'$ | Right tensor $B'$ | Use case |
|---------------|-----------------|------------------|----------|
| `"right"` (default) | $U$ | $S V^\dagger$ | Left-sweep; keeps $A'$ left-canonical |
| `"left"` | $U S$ | $V^\dagger$ | Right-sweep; keeps $B'$ right-canonical |
| `"both"` | $U \sqrt{S}$ | $\sqrt{S} V^\dagger$ | Symmetric split; neither tensor is canonical |

The default in `apply_two_site_gate` is `absorb="right"` — appropriate for a left-to-right TEBD sweep where $A'$ should be left-isometric before the next bond update.

In `from_statevector`, the default is `absorb="right"` as well, producing a right-canonical MPS after the full left-to-right SVD sweep (with the last site holding the norm).

---

## 7. Gate ordering

### Two-site gate matrix convention

A two-site gate $U$ on sites $(i, i+1)$ with local dimension $d$ is stored as a $(d^2 \times d^2)$ matrix or equivalently a $(d, d, d, d)$ rank-4 tensor. The index ordering is:

```
U[i', j', i, j]   — output-left, output-right, input-left, input-right
```

As a matrix: rows index the **output** basis $(i' d + j')$, columns index the **input** basis $(i\,d + j)$.

**Example — CNOT with $d=2$, control = left site:**

```
U_CNOT[i', j', i, j]:
  i'=i  (control unchanged)
  j'=j XOR i  (target flipped if control=1)
```

### Real-time gate from Hamiltonian

`two_site_gate_from_hamiltonian(H, dt)` computes:

$$U = e^{-i\,dt\,H}$$

where $H$ is a $(d^2 \times d^2)$ Hermitian matrix for a local two-site interaction. The result is reshaped to `(d, d, d, d)` with the `(i', j', i, j)` ordering above.

### Imaginary-time gate

`two_site_gate_imaginary(H, dt)` computes:

$$G = e^{-\tau\,H}, \qquad \tau = dt > 0$$

This is **not unitary**. After applying $G$ the MPS must be renormalised (handled automatically by `finite_tebd_imaginary` via `TEBDConfig(normalize=True)`).

---

## 8. MPO structure

### Virtual bond ordering (MPO row/column)

The MPO virtual bond for a standard two-body Hamiltonian is arranged as:

```
W_i = [ I    0    0  ]
      [ O_L  0    0  ]
      [ h    O_R  I  ]
```

where the first row/column corresponds to the **left boundary** (passes the identity through from the left) and the last row/column corresponds to the **right boundary** (accumulates the completed term). This is the standard "zipper" or "finite-state machine" MPO construction.

For the TFIM and Heisenberg MPOs in `hamiltonian/`, the MPO bond dimension is:
- TFIM: $\chi_W = 3$
- Heisenberg / XXZ: $\chi_W = 5$

### `MPO.apply(mps)`

Contracts the MPO with the MPS to produce a new MPS. The result has physical indices in the same ordering as the input MPS. Bond dimensions of the output MPS grow as $\chi_\text{out} = \chi_\text{MPS} \times \chi_W$ before any compression.

### `MPO.to_dense()`

Converts the MPO to a dense $(d^L \times d^L)$ matrix. The row index is the **ket** (output) multi-index and the column index is the **bra** (input) multi-index, both in **little-endian qubit order** (site 0 is the least significant bit). This matches `MPS.to_dense()` so that `mpo.to_dense() @ mps.to_dense()` gives the correct action.

---

## 9. Hamiltonian sign conventions

All Hamiltonians are written so that the **ground state energy is negative** for ferromagnetic/antiferromagnetic ordering in their standard parameter regime.

### TFIM

$$H_\text{TFIM} = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i$$

- $J > 0$: ferromagnetic coupling.
- $h > 0$: transverse field.
- Critical point at $h/J = 1$ for the infinite chain.

### Heisenberg (XXX)

$$H_\text{Heis} = J \sum_i \left(X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}\right)$$

- $J > 0$: antiferromagnetic. Ground state energy per bond $\approx -0.4431$ for the infinite chain.
- $J < 0$: ferromagnetic.

### XX model

$$H_\text{XX} = J \sum_i \left(X_i X_{i+1} + Y_i Y_{i+1}\right)$$

### Pauli matrix convention

We use the standard Pauli matrices (not $\hbar/2$ normalised):

$$X = \begin{pmatrix}0&1\\1&0\end{pmatrix}, \quad Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}, \quad Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$$

The computational basis is $|0\rangle = (1,0)^T$ (spin-up, $+Z$ eigenstate) and $|1\rangle = (0,1)^T$ (spin-down, $-Z$ eigenstate).

---

## 10. Statevector qubit ordering

`MPS.to_dense()` returns a $d^L$-dimensional statevector. The multi-index is **big-endian**: site 0 is the **most significant** position.

For $L=3$, $d=2$:

```
index k  =  σ_0 * 4  +  σ_1 * 2  +  σ_2 * 1
psi[k]   =  <σ_0 σ_1 σ_2 | ψ>
```

So `psi[0]` = $\langle 000|\psi\rangle$, `psi[1]` = $\langle 001|\psi\rangle$, `psi[4]` = $\langle 100|\psi\rangle$, etc.

`MPO.to_dense()` follows the same convention so that matrix-vector products are consistent.

> **Watch out:** some quantum computing libraries (e.g. Qiskit) use little-endian (site 0 = least significant). When comparing with those libraries, reverse the qubit order.

---

## 11. DMRG conventions

### Two-site update

The finite DMRG implemented here uses the **two-site update** at each step:
1. Form the effective Hamiltonian $H_\text{eff}$ for sites $(i, i+1)$ from left/right environments and the MPO.
2. Diagonalise $H_\text{eff}$ (via `scipy.sparse.linalg.eigsh` with `which="SA"`) to get the ground-state two-site tensor $\Theta$.
3. SVD-split $\Theta$ into $A_i$ and $A_{i+1}$ with optional truncation.
4. Absorb singular values to the **right** on the left-to-right sweep, and to the **left** on the right-to-left sweep, maintaining mixed-canonical form.

### Sweep direction

A full DMRG sweep is: left-to-right (sites $0 \to L-2$) then right-to-left (sites $L-2 \to 0$). One full sweep = one DMRG iteration. Convergence is checked by comparing the energy after each full sweep.

### Environment update

Left environment $L_i$ and right environment $R_i$ are updated incrementally:

$$L_{i+1}[a', a] = \sum_{\alpha', \alpha, \sigma} A^*_i[\alpha', \sigma, a']\; W_i[\alpha', \sigma', \sigma, \alpha]\; A_i[\alpha, \sigma, a]\; L_i[\alpha', \alpha]$$

(and analogously for right environments). This is the standard MPS-MPO-MPS contraction. Environments are stored as rank-3 tensors with shape `(χ_mps, χ_mpo, χ_mps)`.

### DMRGConfig

```python
@dataclass
class DMRGConfig:
    n_sweeps: int        # number of full sweeps
    max_bond_dim: int    # maximum bond dimension χ
    svd_cutoff: float    # singular-value cutoff (default 1e-10)
    tol: float           # energy convergence tolerance (default 1e-8)
    verbose: bool        # print energy per sweep (default False)
```

---

## 12. TEBD conventions

### Trotter decomposition

**First-order (Lie–Trotter):**

$$e^{-i\,dt\,H} \approx \prod_{\langle i,i+1\rangle \in \text{even}} e^{-i\,dt\,h_{i,i+1}} \cdot \prod_{\langle i,i+1\rangle \in \text{odd}} e^{-i\,dt\,h_{i,i+1}} + O(dt^2)$$

Even bonds: $(0,1), (2,3), \ldots$ — applied first.  
Odd bonds: $(1,2), (3,4), \ldots$ — applied second.

**Second-order (Strang / Leapfrog):**

$$e^{-i\,dt\,H} \approx \prod_\text{even} e^{-i\,\frac{dt}{2}\,h} \cdot \prod_\text{odd} e^{-i\,dt\,h} \cdot \prod_\text{even} e^{-i\,\frac{dt}{2}\,h} + O(dt^3)$$

`finite_tebd_strang` implements this: half-step even gates → full odd gates → half-step even gates per time step.

### Gate layer ordering

Within each layer, gates are applied **left to right** (bond 0 first, then bond 2, then bond 4, … for even; bond 1, then bond 3, … for odd). This is purely a sweep convention and has no physical significance for commuting gates within a layer.

### TEBDConfig

```python
@dataclass
class TEBDConfig:
    n_steps: int                          # number of Trotter steps
    truncation: TruncationPolicy | None   # SVD truncation per gate application
    normalize: bool                       # renormalize MPS after each full step
    verbose: bool                         # print norm/energy per step
```

---

## 13. Imaginary-time evolution

Imaginary-time TEBD (`finite_tebd_imaginary`) evolves under:

$$|\psi(\tau)\rangle = e^{-\tau H}|\psi(0)\rangle \Big/ \|e^{-\tau H}|\psi(0)\rangle\|$$

with $\tau = n\_steps \times dt$. The gates are **non-unitary** ($G = e^{-dt\,h_{ij}}$, real positive spectrum for $h_{ij}$ positive semi-definite). After each full Trotter step the MPS is renormalised so the norm does not underflow.

**Ground-state preparation recipe:**

1. Start from a random or product MPS that is not orthogonal to the ground state.
2. Choose $dt$ small enough that the first-order Trotter error is acceptable (typically $dt \in [0.01, 0.1]$).
3. Run for enough steps that $\langle E \rangle$ converges to within tolerance.
4. Cross-check final energy against DMRG.

The imaginary-time gates are built by `two_site_gate_imaginary(H_local, dt)` which computes $e^{-dt H_\text{local}}$ via `scipy.linalg.expm` (exact for the small $d^2 \times d^2$ matrix).

---

## 14. Measurement conventions

### `measure_local(mps, operators)`

Computes single-site expectation values $\langle \psi | O_i | \psi \rangle$ for each site $i$ via a left–right transfer-matrix sweep:

1. Build left environments $\{L_i\}$ by contracting MPS tensors left to right.
2. For each site $i$, insert $O_i$ and contract with the matching right environment $R_i$.
3. Normalise by $\langle\psi|\psi\rangle$ (computed from the same sweep).
4. Return `float(Re(...))` — imaginary parts are discarded after a warning if $|\text{Im}| > 10^{-10}$.

`operators` is a `dict[int, np.ndarray]` mapping site index to a $(d \times d)$ operator matrix. Sites not in the dict are skipped.

### Return type

`measure_local` returns `dict[int, float]` — one real number per queried site.

### Operator convention

Operator matrices follow the same basis as the Pauli matrices in §9: row = ket (output), column = bra (input), with $|0\rangle = (1,0)^T$.

---

## 15. Dtype and precision

- Default dtype throughout the library: `np.complex128`.
- All internal contractions are performed in `complex128` even when the state is known to be real, to avoid silent cast errors.
- When computing norms or energies (real scalars), the result is explicitly cast via `float(np.real(...))` — never `float(complex_value)` directly, which raises `ComplexWarning`.
- Singular values are always real and non-negative; they are stored as `float64` arrays.
- `TruncationPolicy.svd_cutoff` defaults to `1e-12` — one order of magnitude below `complex128` machine epsilon ($\approx 2.2 \times 10^{-16}$) relative to the largest singular value.
