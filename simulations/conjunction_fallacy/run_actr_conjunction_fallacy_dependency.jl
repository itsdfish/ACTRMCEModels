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
using ACTRMCEModels: compute_cf_prob
#######################################################################################################################################
#                                                   setup model
#######################################################################################################################################
Random.seed!(23069)
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
    blcs1 = rand(blc_dist, 4)
    blcs2 = deepcopy(blcs1)
    blcs1[2] -= rand(Uniform(0, .5))
    blcs2[2] += rand(Uniform(0, .5))
    δ = rand(δ_dist)
    model1 = ACTRCPI(; blcs = blcs1, Θ..., δ)
    # probability of B
    p_B = judge(model1; B = true)
    # probability of F ∧ B
    p_FB = judge(model1; F = true, B = true)
    cf_prob1 = compute_cf_prob(model1, p_FB, p_B)

    model2 = ACTRCPI(; blcs = blcs2, Θ..., δ)
    # probability of B
    p_B = judge(model2; B = true)
    # probability of F ∧ B
    p_FB = judge(model2; F = true, B = true)
    cf_prob2 = compute_cf_prob(model2, p_FB, p_B)

    return cf_prob1 - cf_prob2
end

n_sim = 10_000
results = map(1:n_sim) do _
    simulate(; Θ, config...)
end

mean(results .> 0)
