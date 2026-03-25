# tests/core/test_utils.py
import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.utils import expectation_value
from tensor_network_library.hamiltonian.models import tfim_mpo, tfim_dense


def random_normalized_state(L: int, *, dtype=np.complex128) -> np.ndarray:
    dim = 2**L
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v.astype(dtype)
    v /= np.linalg.norm(v)
    return v


class TestExpectationValue:
    def test_tfim_expectation_matches_dense(self):
        L = 4
        J, g = 1.0, 0.7

        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        mpo = tfim_mpo(L=L, J=J, g=g)
        H_dense = tfim_dense(L=L, J=J, g=g)

        e_mps = expectation_value(mps, mpo)

        v = mps.to_dense()
        e_dense = float(np.vdot(v, H_dense @ v))

        assert np.allclose(e_mps, e_dense, atol=1e-10)