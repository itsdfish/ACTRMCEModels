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
#######################################################################################################################################
#                                                           setup simulation
#######################################################################################################################################
Θ = (mmp = true, noise = true, s = 0.20, blc = 0.0, τ = -100)

x = map(1:100000) do _
    bl = Normal(0, 0.5)
    δ = rand(Uniform(0, 5))
    n_sub_events1 = rand(1:10)
    n_sub_events2 = rand(1:10)
    model = ACTRUnpacking(; n_sub_events1, n_sub_events2, bl, Θ..., δ)
    judge(model; cause = :cancer) + judge(model; cause = :non_cancer)
end

histogram(x)

Θ = (mmp = false, noise = true, s = 0.20, blc = 0.0)

bl = 0
τ = 1
δ = 1
model = ACTRUnpacking(; n_sub_events1 = 5, n_sub_events2 = 5, bl, Θ..., δ, τ)
judge(model; cause = :cancer) + judge(model; cause = :non_cancer)
