#######################################################################################################################################
#                                                   load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Distributions
using Random
using Revise
using ACTRModels
using ACTRMCEModels
#######################################################################################################################################
#                                                   setup model
#######################################################################################################################################
Random.seed!(8745)
# specify model parameters: partial matching, noise, threshold
Θ = (mmp = true, noise = true, τ = -100, s = 0.20)
config = (;
    δ_dist = Uniform(0, 2),
    blc_dist = Uniform(-2, 2)
)
#######################################################################################################################################
#                                                   run example
#######################################################################################################################################
function simulate(; Θ, δ_dist, blc_dist)
    blcs = rand(blc_dist, 4)
    δ = rand(δ_dist)
    model = ACTRCPI(; blcs, Θ..., δ)
    # probability of B
    p_B = judge(model; B = true)
    # probability of F
    p_F = judge(model; F = true)
    # probability of F ∧ B
    p_FB = judge(model; F = true, B = true)
    return [p_B, p_FB, p_F]
end

n_sim = 10_000
results = map(1:n_sim) do _
    simulate(; Θ, config...)
end

mean(map(x -> x[1] < x[2] < x[3], results))
