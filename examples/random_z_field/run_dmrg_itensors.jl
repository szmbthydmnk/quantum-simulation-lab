# Standalone ITensors DMRG reference for:
#   H1 = -Σ_j h_j Z_j    h_j ~ N(mean=1, std=sqrt(0.1))
#
# Seeds, L, chi_max match run_dmrg.py exactly for direct comparison.
#
# Run:  julia examples/random_z_field/run_dmrg_itensors.jl

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
h = MEAN .+ sqrt(VAR) .* randn(L)

println("[H1-Julia] L=$(L)  chi_max=$(CHI_MAX)")
println("[H1-Julia] h = $(round.(h; digits=4))")

# ---------------------------------------------------------------------------
# Hamiltonian via OpSum
# ITensors S=1/2: Sz = σ_z/2, so multiply by -2*h_j for the full Pauli -h_j Z
# ---------------------------------------------------------------------------
sites = siteinds("S=1/2", L; conserve_qns=false)
os    = OpSum()
for j in 1:L
    os += -2*h[j], "Sz", j
end
H = MPO(os, sites)

# ---------------------------------------------------------------------------
# Exact diagonalisation reference
# ---------------------------------------------------------------------------
function dense_H1(L, h)
    sz = [1.0 0.0; 0.0 -1.0]
    H  = zeros(ComplexF64, 2^L, 2^L)
    for j in 1:L
        op = reshape([1.0+0im], 1, 1)
        for k in 1:L
            op = kron(op, k == j ? -h[j]*sz : Matrix{ComplexF64}(I, 2, 2))
        end
        H .+= op
    end
    return H
end

evals      = sort(real(eigvals(Hermitian(dense_H1(L, h)))))
E_exact    = evals[1]
E_analytic = -sum(abs.(h))
println("[H1-Julia] Exact ground-state energy : $(round(E_exact;    digits=12))")
println("[H1-Julia] Analytic minimum energy   : $(round(E_analytic; digits=12))")

# ---------------------------------------------------------------------------
# Initial MPS
# ---------------------------------------------------------------------------
Random.seed!(INIT_SEED)
psi0 = random_mps(sites; linkdims=CHI_MAX)
E0   = real(inner(psi0', H, psi0))
println("[H1-Julia] Initial bond dims : $(linkdims(psi0))")
println("[H1-Julia] Initial energy    : $(round(E0; digits=12))")

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
println("\n[H1-Julia] Final DMRG energy   : $(round(E_dmrg;     digits=12))")
println("[H1-Julia] Exact energy        : $(round(E_exact;    digits=12))")
println("[H1-Julia] Analytic min energy : $(round(E_analytic; digits=12))")
println("[H1-Julia] |E_dmrg - E_exact|  : $(E_error)")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
csv_path = joinpath(OUT_DIR, "H1_itensors_convergence.csv")
open(csv_path, "w") do io
    writedlm(io, ["sweep" "energy" "max_bond_dim";
                   0       E0       CHI_MAX;
                   MAX_SWEEPS E_dmrg CHI_MAX], ',')
end
println("[H1-Julia] Saved CSV -> $(csv_path)")

p = bar(["Exact", "Analytic", "ITensors DMRG"],
        [E_exact, E_analytic, E_dmrg];
        ylabel="Energy",
        title="H1 = -Σ h_j Z_j  (L=$(L), χ=$(CHI_MAX))",
        legend=false,
        color=[:red, :green, :steelblue])
hline!(p, [E_exact]; linestyle=:dash, color=:red)
plot_path = joinpath(OUT_DIR, "H1_itensors_energy.png")
savefig(p, plot_path)
println("[H1-Julia] Saved plot -> $(plot_path)")
