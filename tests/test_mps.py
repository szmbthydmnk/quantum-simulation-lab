import numpy as np
from tensor_network_library.core.mps import MPS

def test_from_product_state_shapes_and_bonds():
    mps = MPS.from_product_state([0, 1, 0], physical_dim=2)

    assert len(mps) == 3
    # each site tensor should be (1, d, 1)
    for t in mps.tensors:
        assert t.shape == (1, 2, 1)

    # bond dims should be [1, 1, 1, 1] for a product state
    assert mps.bond_dims == [1, 1, 1, 1]

def test_mps_norm_of_product_state():
    mps = MPS.from_product_state([0, 1, 0, 1], physical_dim=2)

    n = mps.norm()
    assert np.isclose(n, 1.0)

    mps.normalize()
    n2 = mps.norm()
    assert np.isclose(n2, 1.0)
