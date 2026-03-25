# ITensors DMRG reference for:
#   H3 = Jz * Σ_{i} Z_i Z_{i+1} - h * Σ_i Z_i
#
# Parameters match run_dmrg.py exactly for direct comparison.
#
# Run:  julia examples/zz_plus_z/run_dmrg_itensors.jl

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
const L          = 20
const CHI_MAX    = 32
const JZ         = 1.0
const H_FIELD    = 0.5
const INIT_SEED  = 11
const MAX_SWEEPS = 20
const ENERGY_TOL = 1e-10

OUT_DIR = joinpath(@__DIR__, "results")
mkpath(OUT_DIR)

println("[H3-Julia] L=$(L)  chi_max=$(CHI_MAX)  Jz=$(JZ)  h=$(H_FIELD)")

# ---------------------------------------------------------------------------
# Hamiltonian via OpSum
#
# ITensors S=1/2: Sz eigenvalues are ±1/2, so the Pauli Z = 2*Sz.
# H3 = Jz Σ Z_i Z_{i+1} - h Σ Z_i
#    = 4*Jz Σ Sz_i Sz_{i+1} - 2*h Σ Sz_i
# ---------------------------------------------------------------------------
sites = siteinds("S=1/2", L; conserve_qns=false)
os    = OpSum()
for j in 1:(L - 1)
    os += 4 * JZ, "Sz", j, "Sz", j + 1
end
for j in 1:L
    os += -2 * H_FIELD, "Sz", j
end
H = MPO(os, sites)

# ---------------------------------------------------------------------------
# Exact dense reference  (only practical for L ≲ 16; skip for large L)
# ---------------------------------------------------------------------------
function dense_H3(L, Jz, h)
    sz  = [1.0 0.0; 0.0 -1.0]   # Pauli Z
    id2 = Matrix{ComplexF64}(I, 2, 2)
    dim = 2^L
    H   = zeros(ComplexF64, dim, dim)
    # ZZ terms
    for j in 1:(L - 1)
        op = reshape([1.0+0im], 1, 1)
        for k in 1:L
            local_op = (k == j || k == j + 1) ? Jz * sz : id2
            # For the two-site term we need to compose carefully:
            op = k == j ? kron(reshape([1.0+0im], 1, 1), sz) :
                 k == j + 1 ? kron(op, sz) : kron(op, id2)
        end
        H .+= Jz .* op
    end
    return H
end

if L <= 16
    # Build a simple dense H3 without the helper above (which has a bug
    # for non-adjacent sites) using direct Kronecker products.
    sz   = ComplexF64[1 0; 0 -1]
    id2  = Matrix{ComplexF64}(I, 2, 2)
    dim  = 2^L
    H_dense = zeros(ComplexF64, dim, dim)
    for j in 1:(L - 1)
        left  = j == 1 ? sz : Matrix{ComplexF64}(I, 2^(j-1), 2^(j-1))
        if j > 1
            left = kron(Matrix{ComplexF64}(I, 2^(j-1), 2^(j-1)), sz)
        else
            left = sz
        end
        # Correct kron: I^(j-1) ⊗ Z_j ⊗ Z_{j+1} ⊗ I^(L-j-1)
        op = kron(
            Matrix{ComplexF64}(I, 2^(j-1), 2^(j-1)),
            kron(sz, kron(sz, Matrix{ComplexF64}(I, 2^(L-j-1), 2^(L-j-1))))
        )
        H_dense .+= JZ .* op
    end
    for j in 1:L
        op = kron(
            Matrix{ComplexF64}(I, 2^(j-1), 2^(j-1)),
            kron(sz, Matrix{ComplexF64}(I, 2^(L-j), 2^(L-j)))
        )
        H_dense .+= (-H_FIELD) .* op
    end
    evals  = sort(real(eigvals(Hermitian(H_dense))))
    E_exact = evals[1]
    println("[H3-Julia] Exact ground-state energy: $(round(E_exact; digits=12))")
else
    E_exact = NaN
    println("[H3-Julia] L=$(L) > 16: skipping dense exact diagonalisation")
end

# ---------------------------------------------------------------------------
# Initial MPS
# ---------------------------------------------------------------------------
Random.seed!(INIT_SEED)
psi0 = random_mps(sites; linkdims=CHI_MAX)
E0   = real(inner(psi0', H, psi0))
println("[H3-Julia] Initial bond dims : $(linkdims(psi0))")
println("[H3-Julia] Initial energy    : $(round(E0; digits=12))")

# ---------------------------------------------------------------------------
# DMRG
# ---------------------------------------------------------------------------
sweeps = Sweeps(MAX_SWEEPS)
maxdim!(sweeps, CHI_MAX)
cutoff!(sweeps, ENERGY_TOL)
noise!(sweeps, 0.0)

energy, psi = dmrg(H, psi0, sweeps; outputlevel=1, ishermitian=true)

E_dmrg = real(energy)
println("\n[H3-Julia] Final DMRG energy : $(round(E_dmrg; digits=12))")
if !isnan(E_exact)
    println("[H3-Julia] Exact energy      : $(round(E_exact; digits=12))")
    println("[H3-Julia] |E_dmrg-E_exact| : $(abs(E_dmrg - E_exact))")
end
println("[H3-Julia] Final bond dims   : $(linkdims(psi))")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
csv_path = joinpath(OUT_DIR, "H3_itensors_convergence.csv")
open(csv_path, "w") do io
    if !isnan(E_exact)
        writedlm(io, ["label" "energy";
                       "initial" E0;
                       "dmrg" E_dmrg;
                       "exact" E_exact], ',')
    else
        writedlm(io, ["label" "energy";
                       "initial" E0;
                       "dmrg" E_dmrg], ',')
    end
end
println("[H3-Julia] Saved CSV -> $(csv_path)")

labels = isnan(E_exact) ? ["Initial", "ITensors DMRG"] : ["Initial", "ITensors DMRG", "Exact"]
vals   = isnan(E_exact) ? [E0, E_dmrg] : [E0, E_dmrg, E_exact]
p = bar(labels, vals;
        ylabel="Energy",
        title="H3 = Jz Σ ZZ - h Σ Z  (L=$(L), χ=$(CHI_MAX))",
        legend=false,
        color=[:grey, :steelblue, :red])
if !isnan(E_exact)
    hline!(p, [E_exact]; linestyle=:dash, color=:red, label="Exact")
end
hline!(p, [E_dmrg]; linestyle=:dot, color=:steelblue)
plot_path = joinpath(OUT_DIR, "H3_itensors_energy.png")
savefig(p, plot_path)
println("[H3-Julia] Saved plot -> $(plot_path)")
