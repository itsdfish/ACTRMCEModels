#######################################################################################################################################

#                                                           load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using ACTRModels
using Distributions
using ACTRMCEModels
using LaTeXStrings
using Plots
using Random
Random.seed!(87544)

Θ = (sa = true, noise = true, γ = 3.0, s = 0.20, blc = 0.0, τ = -1000)

lb = 0
ub = 6
μ = 0
σ = 0.5
config2 = (
    n_ch = 4,
    n_ch̄ = 1,
    n_gh = 1,
    n_gh̄ = 4,
    bl_ch = 0.0,
    bl_ch̄ = 0.0,
    bl_gh = 0.0,
    bl_gh̄ = 0.0
)
model = ACTROrderEffect(; config2..., Θ...)
order_effects = compute_order_effects(model)
q1 = compute_q1(model)
println("q1 $(round(q1, digits = 3)) order effects $(round.(order_effects, digits = 3))")
