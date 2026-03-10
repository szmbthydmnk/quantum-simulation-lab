"""
DMRG (Density Matrix Renormalization Group) algorithm.

Uses Schollwöck's mixed-canonical form: sweep left-to-right and right-to-left,
solving two-site optimization problems at each step.

Key steps:
1. Start with an MPS in left-canonical form (center=0).
2. Sweep right: optimize two-site object (sites i, i+1), left-normalize, move center.
3. At rightmost site, sweep left: optimize two-site object, right-normalize, move center.
4. Repeat until convergence.
"""

