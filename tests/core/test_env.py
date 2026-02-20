from tensor_network_library.core.env import Environment

def test_environment_basic_fields():
    env = Environment(system_type="spin-1/2", L=10, d=2, bc="open", max_bond_dim=128, truncation_tol=1e-10)

    assert env.system_type == "spin-1/2"
    assert env.L == 10
    assert env.d == 2
    assert env.bc == "open"
    assert env.max_bond_dim == 128
    assert env.truncation_tol == 1e-10
