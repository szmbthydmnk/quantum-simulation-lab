# Standalone ITensors DMRG reference for:
#   H2 = Σ_j J_j X_j    J_j ~ N(mean=1, std=sqrt(0.1))
#
# Seeds, L, chi_max match run_dmrg.py exactly for direct comparison.
#
# Run:  julia examples/random_x_field/run_dmrg_itensors.jl

import Pkg
for pkg in ["ITensors", "ITensorMPS", "Plots"]
    try
        @eval using $(Symbol(pkg))
    catch
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

using Random, LinearAlgebra, DelimitedFiles

# ---------------------------------------------------------------------------
# Parameters  (match run_dmrg.py)
# ---------------------------------------------------------------------------
const L          = 10
const CHI_MAX    = 4
const MEAN       = 1.0
const VAR        = 0.1
const SEED       = 7
const INIT_SEED  = 99
const MAX_SWEEPS = 20
const ENERGY_TOL = 1e-10

OUT_DIR = joinpath(@__DIR__, "results")
mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# Couplings
# ---------------------------------------------------------------------------
Random.seed!(SEED)
J = MEAN .+ sqrt(VAR) .* randn(L)

println("[H2-Julia] L=$(L)  chi_max=$(CHI_MAX)")
println("[H2-Julia] J = $(round.(J; digits=4))")

# ---------------------------------------------------------------------------
# Hamiltonian via OpSum
# ITensors S=1/2: Sx = σ_x/2, so multiply by 2 to get the full Pauli X
# ---------------------------------------------------------------------------
sites = siteinds("S=1/2", L; conserve_qns=false)
os    = OpSum()
for j in 1:L
    os += 2*J[j], "Sx", j
end
H = MPO(os, sites)

# ---------------------------------------------------------------------------
# Exact diagonalisation reference
# ---------------------------------------------------------------------------
function dense_H2(L, J)
    sx = [0.0 1.0; 1.0 0.0]
    H  = zeros(ComplexF64, 2^L, 2^L)
    for j in 1:L
        op = reshape([1.0+0im], 1, 1)
        for k in 1:L
            op = kron(op, k == j ? J[j]*sx : Matrix{ComplexF64}(I, 2, 2))
        end
        H .+= op
    end
    return H
end

evals      = sort(real(eigvals(Hermitian(dense_H2(L, J)))))
E_exact    = evals[1]
E_analytic = -sum(abs.(J))
println("[H2-Julia] Exact ground-state energy : $(round(E_exact;    digits=12))")
println("[H2-Julia] Analytic minimum energy   : $(round(E_analytic; digits=12))")

# ---------------------------------------------------------------------------
# Initial MPS
# ---------------------------------------------------------------------------
Random.seed!(INIT_SEED)
psi0 = random_mps(sites; linkdims=CHI_MAX)
E0   = real(inner(psi0', H, psi0))
println("[H2-Julia] Initial bond dims : $(linkdims(psi0))")
println("[H2-Julia] Initial energy    : $(round(E0; digits=12))")

# ---------------------------------------------------------------------------
# DMRG
# ---------------------------------------------------------------------------
sweeps = Sweeps(MAX_SWEEPS)
maxdim!(sweeps, CHI_MAX)
cutoff!(sweeps, ENERGY_TOL)
noise!(sweeps, 0.0)

energy, psi = dmrg(H, psi0, sweeps; outputlevel=1, ishermitian=true)

E_dmrg  = real(energy)
E_error = abs(E_dmrg - E_exact)
println("\n[H2-Julia] Final DMRG energy   : $(round(E_dmrg;     digits=12))")
println("[H2-Julia] Exact energy        : $(round(E_exact;    digits=12))")
println("[H2-Julia] Analytic min energy : $(round(E_analytic; digits=12))")
println("[H2-Julia] |E_dmrg - E_exact|  : $(E_error)")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
csv_path = joinpath(OUT_DIR, "H2_itensors_convergence.csv")
open(csv_path, "w") do io
    writedlm(io, ["sweep" "energy" "max_bond_dim";
                   0       E0       CHI_MAX;
                   MAX_SWEEPS E_dmrg CHI_MAX], ',')
end
println("[H2-Julia] Saved CSV -> $(csv_path)")

p = bar(["Exact", "Analytic", "ITensors DMRG"],
        [E_exact, E_analytic, E_dmrg];
        ylabel="Energy",
        title="H2 = Σ J_j X_j  (L=$(L), χ=$(CHI_MAX))",
        legend=false,
        color=[:red, :green, :steelblue])
hline!(p, [E_exact]; linestyle=:dash, color=:red)
plot_path = joinpath(OUT_DIR, "H2_itensors_energy.png")
savefig(p, plot_path)
println("[H2-Julia] Saved plot -> $(plot_path)")
