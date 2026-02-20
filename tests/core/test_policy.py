import pytest
import numpy as np
from tensor_network_library.core.policy import TruncationPolicy

def test_truncation_policy_respects_cutoff_and_max_bond_dim():
    s = np.array([0.9, 0.5, 0.1, 0.01])  # singular values
    policy = TruncationPolicy(max_bond_dim=3, cutoff=0.05)

    chi = policy.choose_bond_dim(s)

    # s^2 = [0.81, 0.25, 0.01, 0.0001]; with cutoff=0.05, last two are below
    # so by tolerance we only need 2, but max_bond_dim=3, so chi=2
    assert chi == 2

def test_truncation_policy_hits_max_bond_dim():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    policy = TruncationPolicy(max_bond_dim=2, cutoff=0.0)

    chi = policy.choose_bond_dim(s)
    # tolerance allows all, but max_bond_dim=2, so chi=2
    assert chi == 2

def test_truncation_policy_non_strict_caps_at_max():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    policy = TruncationPolicy(max_bond_dim=2, cutoff=0.0, strict=False)

    chi = policy.choose_bond_dim(s)
    assert chi == 2  # capped at max_bond_dim

def test_truncation_policy_strict_raises_when_needed_exceeds_max():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    policy = TruncationPolicy(max_bond_dim=2, cutoff=0.0, strict=True)

    with pytest.raises(ValueError):
        _ = policy.choose_bond_dim(s)