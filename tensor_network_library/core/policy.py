from dataclasses import dataclass

@dataclass
class TruncationPolicy:
    max_bond_dimension: int     #
    cutoff: float               # discrad singular values with value smaller than sqrt(cutoff)
    
    def choose_bond_dim(self, singular_values) -> int:
        s_sqr = singular_values ** 2
        
        keep = (s_sqr >= self.cutoff).sum()
        return min(keep, self.max_bond_dimension)