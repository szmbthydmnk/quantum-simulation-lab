from dataclasses import dataclass   # This module provides a decorator and functions for automatically adding generated special methods such as __init__() and __repr__() to user-defined classes. It was originally described in PEP 557.
from typing import Literal      # variable init. with Literal cannot have values outside of the provided values

boundary_condition = Literal["open", "periodic"]

@dataclass
class Environment:
    system_type: str                    # e.g. spin-1/2 or qudit
    L: int                              # number of sites
    d: int                              # local dimension
    bc: boundary_condition = "open"     # boundary condition
    max_bond_dim: int = 64              # I don't want to have a dynamical bond dimension in this build yet.
    truncation_tol: float = 1e-10       # 
    complex_dtype: type = complex       #