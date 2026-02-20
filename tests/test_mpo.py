import numpy as np

from tensor_network_library.core.mps import MPS
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

def test_identity_mpo_applies_as_identity_on_mps():
    state = [0, 1, 0]
    mps = MPS.from_product_state(state, physical_dims = 2)
    mpo = MPO.identity(len(state), physical_dims = 2)

    new_mps = mpo.apply(mps)

    # bond dims may change (1 -> 1 still here), but state vector should be identical
    # For small L, we can reconstruct the full state brute-force.
    def to_statevector(mps_obj):
        # contract chain explicitly
        psi = mps_obj.tensors[0].data[0, :, 0]  # shape (d,)
        for t in mps_obj.tensors[1:]:
            # t: (1, d, 1), psi: (..., d_prev)
            psi = np.tensordot(psi, t.data[0, :, 0], axes=0)  # Kronecker product
        return psi.reshape(-1)

    psi_original = to_statevector(mps)
    psi_new = to_statevector(new_mps)

    assert np.allclose(psi_original, psi_new)
    
def test_mpo_apply_matches_dense():
    L = 3
    d = 2
    state = [1, 1, 0]

    mps = MPS.from_product_state(state, physical_dims = d)
    mpo = MPO.identity(L, physical_dims = d)  # later you can plug in nontrivial MPO

    # Apply with your TN code
    mps2 = mpo.apply(mps)

    # Dense reference
    psi = mps.to_dense()
    O = mpo.to_dense()
    psi2_dense = O @ psi

    # Compare to dense version of mps2
    psi2_from_mps = mps2.to_dense()

    assert np.allclose(psi2_dense, psi2_from_mps)