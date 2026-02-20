import numpy as np
from tensor_network_library.core.mps import MPS
from tensor_network_library.core.mpo import MPO
from tensor_network_library.core.canonical import left_canonicalize

def test_left_canonical_preserves_state():
    state = [0, 1, 0]
    mps = MPS.from_product_state(state, physical_dims=2)
    mpo = MPO.identity(len(state), physical_dims=2)
    mps = mpo.apply(mps)  # make it slightly less trivial if you want

    psi_before = mps.to_dense()
    left_canonicalize(mps)
    psi_after = mps.to_dense()

    assert np.allclose(psi_before, psi_after)
