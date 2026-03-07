"""Canonical form transformaitons for MPS."""

import numpy as np
from typing import Optional
from .mps import MPS
from .policy import TruncationPolicy  # not used yet, but later
from .tensor import Tensor

def left_canonicalize(mps: MPS, policy: Optional[TruncationPolicy] = None) -> MPS:
    """
    Bring MPS into left-canonical form.
    
    In left-canonical form, each tensor A satisfies:
    Σ_{s,α} A^s_{\alpha,\beta} A^{s*}_{\alpha,\beta'} = \delta_{\beta,\beta'}
    
    Arguments:
        mps: MPS to canonicalize
        policy: optional variable to set max bond dimension and other policy set bounds
        
    Returns:
        Left-canonical MPS
    """
    
    mps = mps.copy()
    
    L = len(mps.tensors)
    
    for i in range(L - 1):
        # Get current tensor
        A = mps.tensors[i]
        
        # perform the QR decomposition
        Q, R = A.qr_decomposition(left_indices=[0, 1], right_indices=[2])
            # The shape of Q: (chi_left, d, chi_prime)
            # The shape of R: (chi_prime, chi_right)
        
        # Now the new first MPS is the Q tensor:
        mps.tensors[i] = Q
        
        # We need to "merge R with the rest of the MPS":
        next_tensor = mps.tensors[i + 1]
        temp = Tensor(np.tensordot(R.data, next_tensor.data, axes=([1], [0])))
        
        mps.tensors[i + 1] = temp
    
    return mps
