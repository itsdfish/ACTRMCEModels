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
#######################################################################################################################################
#                                                           simulation 1
#######################################################################################################################################
n_sim = 10_000
Θ = (sa = true, noise = true, s = 0.20, blc = 0.0, τ = -1000)

lb = 0
ub = 6
μ = 0
σ = 0
config1 = (
    n_ch = DiscreteUniform(lb, ub),
    n_ch̄ = DiscreteUniform(lb, ub),
    n_gh = DiscreteUniform(lb, ub),
    n_gh̄ = DiscreteUniform(lb, ub),
    bl_ch = Normal(μ, σ),
    bl_ch̄ = Normal(μ, σ),
    bl_gh = Normal(μ, σ),
    bl_gh̄ = Normal(μ, σ),
    γ = Uniform(2, 5)
)

qqs1 = map(1:n_sim) do _
    model = ACTROrderEffect(; config1..., Θ...)
    return compute_q1(model)
end
#######################################################################################################################################
#                                                           simulation 2
#######################################################################################################################################
lb = 0
ub = 6
μ = 0
σ = 0.5
config2 = (
    n_ch = DiscreteUniform(lb, ub),
    n_ch̄ = DiscreteUniform(lb, ub),
    n_gh = DiscreteUniform(lb, ub),
    n_gh̄ = DiscreteUniform(lb, ub),
    bl_ch = Normal(μ, σ),
    bl_ch̄ = Normal(μ, σ),
    bl_gh = Normal(μ, σ),
    bl_gh̄ = Normal(μ, σ),
    γ = Uniform(2, 5)
)
qqs2 = map(1:n_sim) do _
    model = ACTROrderEffect(; config2..., Θ...)
    return compute_q1(model)
end
#######################################################################################################################################
#                                                           plot results
#######################################################################################################################################
pyplot()

histogram(
    qqs1,
    xlims = (-0.5, 0.5),
    ylims = (0, 20),
    xlabel = L"q",
    ylabel = "Density",
    axis = font(7),
    norm = true,
    grid = false,
    label = L"\sigma_{\beta} = 0",
    legendfontsize = 5,
    bins = 75,
    alpha = 0.7,
    size = (230, 150),
    dpi = 300
)

histogram!(
    qqs2,
    color = :darkred,
    label = L"\sigma_{\beta} = .50",
    legendfontsize = 5,
    norm = true,
    grid = false,
    alpha = 0.7
)
savefig("actr_q_distribution.png")
