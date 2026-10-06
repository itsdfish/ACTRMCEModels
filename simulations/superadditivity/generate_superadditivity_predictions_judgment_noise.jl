#######################################################################################################################################
#                                                           load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using ACTRModels
using ACTRMCEModels
using Distributions
using Plots
using Random
using ACTRMCEModels: simulate_trial
using ACTRMCEModels: set
include("simulation_utilities.jl")
Random.seed!(2003)
#######################################################################################################################################
#                                                           setup simulation
#######################################################################################################################################
Θ = (mmp = true, noise = true, s = 0.20, blc = 0.0, n_retrievals = 10)
n_sim = 1000
n_subj = 1000
#######################################################################################################################################
#                                                           run simulation
#######################################################################################################################################
config = (;
    τ_dist = Normal(0, 1),
    sub_event_dist = DiscreteUniform(1, 5),
    δ_dist = Uniform(0, 2),
    blc_dist = Normal(0, 0.50),
    response_func = respond
)

results = simulate_group(n_sim, n_subj; Θ, config...)
pooled_ufs = vcat(results...)
# proportion of subjects whose mean unpacking factors are less than one
mean(mean(results) .< 1)
# proportion of superadditive judgments pooled across subjects 
mean(pooled_ufs .< 1)
# boot strap 50 subjects
mean(map(_ -> mean_unpacking_factors(50; Θ, config...), 1:10_000) .< 1)
histogram(pooled_ufs, norm = true, leg = false, grid = false)
