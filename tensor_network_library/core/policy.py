from dataclasses import dataclass
import numpy as np 

@dataclass
class TruncationPolicy:
    """
    Decide how many singular values to keep, based on the cutoff parameter and max_bond_dim.
        
    Attributes:
        cutoff: discard singular values that square is smaller than cutoff
        ma_bond_dim: upper bound on kept singular values.
        strict: 
            - False: obey max_bond_dim even if tolerance would require more.
            - True: raise if tolerance requires mor than max_bond_dim    
    """
    max_bond_dim: int           #
    cutoff: float               # discrad singular values with value smaller than sqrt(cutoff)
    strict: bool = False        #
    
    def choose_bond_dim(self, singular_values: np.ndarray) -> int:
        s = np.asarray(singular_values, dtype = float)
        s_sqr = s ** 2
        
        # How many satisfy tolerance:
        keep_tol = int((s_sqr >= self.cutoff).sum())
        
        if keep_tol > self.max_bond_dim:
            if self.strict:
                raise ValueError(
                    f"TruncationPolicy strict violation: "
                    f"needed {keep_tol} singular values to satisfy cutoff={self.cutoff}, "
                    f"but max_bond_dim={self.max_bond_dim}."
                )
            
            return self.max_bond_dim

        return keep_tol
        
                
                
                