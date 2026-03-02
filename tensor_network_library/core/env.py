from dataclasses import dataclass   # This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes. It was originally described in PEP 557.
from typing import Literal      # variable init. with Literal cannot have values outside of the provided values

boundary_condition = Literal["open", "periodic"]
system_type = Literal["qubit", "spin-1/2"]          # qubit and spin-1/2 will be identical for now, later on I would like to implement a fermionic distinction between the two as well
                                                    # as qudit like qudit-# where # would be the dimension of the local hilbert space.
@dataclass
class Environment:
    L: int                              # number of sites
    d: int                              # local dimension
    
    system_type: str = "qubit"          # e.g. spin-1/2 or qudit
    truncation_tol: float = 1e-10       # 
    bc: boundary_condition = "open"     # boundary condition
    max_bond_dim: int = 64              # I don't want to have a dynamical bond dimension in this build yet.
    complex_dtype: type = complex       #

