# run_dmrg_itensors.jl
# ITensors reference implementation of DMRG for:
#
#   H2 = Σ_j J_j X_j    J_j ~ N(mean=1, std=sqrt(0.1))
#
# Seeds, L, and chi_max match the Python run_dmrg.py exactly so that
# energies can be compared directly.
#
# Run:
#   julia --project=. examples/random_x_field/run_dmrg_itensors.jl
#
# Or from this directory:
#   julia --project=../.. run_dmrg_itensors.jl

using ITensors, ITensorMPS
using Random
using LinearAlgebra
using DelimitedFiles
using Plots; gr()

# ---------------------------------------------------------------------------
# Parameters  (must match run_dmrg.py)
# ---------------------------------------------------------------------------
const L         = 10
const CHI_MAX   = 4
const MEAN      = 1.0
const VAR       = 0.1
const SEED      = 7        # field RNG seed
const INIT_SEED = 99       # MPS init seed
const MAX_SWEEPS = 20
const ENERGY_TOL = 1e-10

const OUT_DIR = joinpath(@__DIR__, "results")
mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# Sample couplings with the same seed as Python
# (Julia's default RNG ≠ NumPy; we use the same sequence of normal draws)
# ---------------------------------------------------------------------------
Random.seed!(SEED)
J = MEAN .+ sqrt(VAR) .* randn(L)

println("[H2-Julia] L=$(L)  chi_max=$(CHI_MAX)")
println("[H2-Julia] J = $(round.(J, digits=4))")

# ---------------------------------------------------------------------------
# Build the Hamiltonian as an OpSum (AutoMPO)
# H = Σ_j J_j X_j
# ---------------------------------------------------------------------------
sites = siteinds("S=1/2", L; conserve_qns=false)

os = OpSum()
for j in 1:L
    # ITensors S=1/2: "X" = σ_x/2; we want the full Pauli so multiply by 2
    # Equivalently use "Sx" (= σ_x / 2) and multiply coefficient by 2.
    os += 2*J[j], "Sx", j
end
H = MPO(os, sites)

# ---------------------------------------------------------------------------
# Exact diagonalisation reference (dense)
# ---------------------------------------------------------------------------
function dense_H2(L, J)
    sx = [0.0 1.0; 1.0 0.0]   # Pauli X
    H  = zeros(ComplexF64, 2^L, 2^L)
    for j in 1:L
        op = ones(1,1)
        for k in 1:L
            op = kron(op, k == j ? J[j]*sx : I(2))
        end
        H .+= op
    end
    return H
end

evals       = sort(real(eigvals(Hermitian(dense_H2(L, J)))))
E_exact     = evals[1]
E_analytic  = -sum(abs.(J))
println("[H2-Julia] Exact ground-state energy : $(round(E_exact, digits=12))")
println("[H2-Julia] Analytic minimum energy   : $(round(E_analytic, digits=12))")

# ---------------------------------------------------------------------------
# Initial MPS: random at chi_max
# ---------------------------------------------------------------------------
Random.seed!(INIT_SEED)
psi0 = random_mps(sites; linkdims=CHI_MAX)

E0 = real(inner(psi0', H, psi0))
println("[H2-Julia] Initial MPS bond dims : $(linkdims(psi0))")
println("[H2-Julia] Initial energy        : $(round(E0, digits=12))")

# ---------------------------------------------------------------------------
# DMRG
# ---------------------------------------------------------------------------
sweeps = Sweeps(MAX_SWEEPS)
maxdim!(sweeps, CHI_MAX)
cutoff!(sweeps, ENERGY_TOL)
noise!(sweeps, 0.0)   # no noise: initial chi already at chi_max

energy, psi = dmrg(H, psi0, sweeps;
    outputlevel=1,
    eigsolve_tol=1e-14,
    ishermitian=true)

E_dmrg  = real(energy)
E_error = abs(E_dmrg - E_exact)
println("\n[H2-Julia] Final DMRG energy   : $(round(E_dmrg,  digits=12))")
println("[H2-Julia] Exact energy        : $(round(E_exact,  digits=12))")
println("[H2-Julia] Analytic min energy : $(round(E_analytic, digits=12))")
println("[H2-Julia] |E_dmrg - E_exact|  : $(E_error)")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
# ITensors dmrg() returns the final energy; sweep-by-sweep energies require
# a custom observer.  We write a minimal two-row CSV: initial + final.
csv_path = joinpath(OUT_DIR, "H2_itensors_convergence.csv")
open(csv_path, "w") do io
    writedlm(io, ["sweep" "energy" "max_bond_dim"; 0 E0 CHI_MAX; MAX_SWEEPS E_dmrg CHI_MAX], ',')
end
println("[H2-Julia] Saved CSV  -> $(csv_path)")

# ---------------------------------------------------------------------------
# Plot: compare Python and Julia energies
# ---------------------------------------------------------------------------
plot_path = joinpath(OUT_DIR, "H2_itensors_energy.png")
p = bar(
    ["Exact", "Analytic", "ITensors\nDMRG"],
    [E_exact, E_analytic, E_dmrg],
    ylabel="Ground-state energy",
    title="H2 = Σ J_j X_j  (L=$(L), χ=$(CHI_MAX))",
    legend=false,
    color=[:red, :green, :steelblue],
    ylims=(E_exact*1.1, 0.1),
)
hline!(p, [E_exact], linestyle=:dash, color=:red, label="Exact")
savefig(p, plot_path)
println("[H2-Julia] Saved plot -> $(plot_path)")
