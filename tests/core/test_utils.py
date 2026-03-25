import numpy as np

from tensor_network_library.core.mps import MPS
from tensor_network_library.core.utils import expectation_value, expectation_value_env
from tensor_network_library.hamiltonian.models import tfim_mpo, heisenberg_mpo


def random_normalized_state(L: int, *, dtype=np.complex128) -> np.ndarray:
    dim = 2 ** L
    v = np.random.randn(dim) + 1j * np.random.randn(dim)
    v = v.astype(dtype)
    v /= np.linalg.norm(v)
    return v


class TestUtilsExpectation:
    def test_tfim_fast_matches_dense(self):
        L = 4
        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        mpo = tfim_mpo(L=L, J=1.0, g=0.7, dtype=np.complex128)

        e_slow = expectation_value(mps, mpo)
        e_fast = expectation_value_env(mps, mpo)

        assert np.allclose(e_fast, e_slow, atol=1e-10)

    def test_heisenberg_fast_matches_dense(self):
        L = 4
        psi = random_normalized_state(L)
        mps = MPS.from_statevector(psi, physical_dims=2, normalize=True)

        mpo = heisenberg_mpo(L=L, Jx=1.0, Jy=0.9, Jz=1.1, h=0.3, dtype=np.complex128)

        e_slow = expectation_value(mps, mpo)
        e_fast = expectation_value_env(mps, mpo)

        assert np.allclose(e_fast, e_slow, atol=1e-10)
