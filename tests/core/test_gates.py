# tests/core/test_apply_two_site_gate.py

from __future__ import annotations

import numpy as np
import pytest

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.gates import apply_two_site_gate
from tensor_network_library.core.policy import TruncationPolicy


def _random_two_site_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a Haar-random unitary on d^2 dim via QR.
    """
    dim = d * d
    X = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(X)
    # Normalize diagonal of R to make Q unitary
    phases = np.diag(R) / np.abs(np.diag(R))
    Q = Q * (phases.conj()[None, :])
    return Q  # shape (d^2, d^2)


@pytest.mark.parametrize("L,site", [(2, 0), (3, 0), (3, 1)])
def test_two_site_gate_matches_dense(L: int, site: int) -> None:
    """
    For small chains, applying U via apply_two_site_gate should match
    dense evolution up to a global phase.
    """
    d = 2
    rng = np.random.default_rng(1234)

    # Random initial MPS via statevector to have exact representation
    dim = d**L
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi = psi / np.linalg.norm(psi)

    mps = MPS.from_statevector(psi, physical_dims=d, name="test_mps", truncation=None)

    U = _random_two_site_unitary(d, rng)

    # Dense evolution
    # Reshape psi to (d,)*L, permute axes so that (site, site+1) are the first two,
    # apply gate, and reshape back.
    psi_tensor = psi.reshape((d,) * L)
    axes = list(range(L))
    axes[0], axes[site] = axes[site], axes[0]
    axes[1], axes[site + 1] = axes[site + 1], axes[1]
    psi_perm = np.transpose(psi_tensor, axes=axes)
    psi_pair = psi_perm.reshape(d * d, -1)  # (d^2, rest)
    psi_pair = U @ psi_pair
    psi_perm2 = psi_pair.reshape((d, d) + psi_perm.shape[2:])
    # Invert permutation
    inv_axes = [0] * L
    for k, a in enumerate(axes):
        inv_axes[a] = k
    psi_tensor2 = np.transpose(psi_perm2, axes=inv_axes)
    psi2_dense = psi_tensor2.reshape(-1)

    # MPS evolution
    mps2 = apply_two_site_gate(mps, U, i=site, truncation=None, absorb="right", inplace=False)
    psi2_mps = mps2.to_dense()

    # Compare up to global phase
    # Find a component with non-negligible amplitude to determine phase
    nz = np.where(np.abs(psi2_dense) > 1e-12)[0]
    if len(nz) == 0:
        # Degenerate case, but then both should be essentially zero
        assert np.allclose(psi2_mps, psi2_dense)
        return

    k0 = int(nz[0])
    phase = psi2_mps[k0] / psi2_dense[k0]
    assert np.allclose(psi2_mps, phase * psi2_dense, atol=1e-10)


def test_two_site_gate_preserves_norm_unitary() -> None:
    """
    For a unitary gate and no truncation, the norm of the MPS must be preserved.
    """
    L = 4
    d = 2
    rng = np.random.default_rng(42)

    mps = MPS.from_random(L=L, chi_max=4, physical_dims=d, seed=0, name="rand_mps")
    n0 = mps.norm()

    U = _random_two_site_unitary(d, rng)
    mps2 = apply_two_site_gate(mps, U, i=1, truncation=None, absorb="right", inplace=False)

    n1 = mps2.norm()
    assert np.isclose(n0, n1, atol=1e-12)


def test_two_site_gate_with_truncation_reduces_bond_dim() -> None:
    """
    If a truncation policy is given, the new bond dimension should not
    exceed truncation.max_bond_dim.
    """
    L = 4
    d = 2
    rng = np.random.default_rng(7)

    mps = MPS.from_random(L=L, chi_max=8, physical_dims=d, seed=1, name="rand_mps_trunc")
    trunc = TruncationPolicy(max_bond_dim=3, cutoff=0.0)

    U = _random_two_site_unitary(d, rng)
    mps2 = apply_two_site_gate(mps, U, i=1, truncation=trunc, absorb="right", inplace=False)

    # Bond dimension between sites 1 and 2 is bond_dims[2]
    assert mps2.bond_dims[2] <= 3