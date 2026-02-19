from tensor_network_library.core.mpo import MPO

def test_identity_mpo_shapes_and_bonds():
    L = 10
    d = 3
    mpo = MPO.identity(L, physical_dims = d)

    assert len(mpo) == L

    for t in mpo.tensors:
        assert t.shape == (1, d, d, 1)

    assert mpo.bond_dims == [1] * (L + 1)
    assert mpo.physical_dims == [d] * L
