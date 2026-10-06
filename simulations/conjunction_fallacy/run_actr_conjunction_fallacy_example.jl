#######################################################################################################################################
#                                                   load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using ACTRModels
using DataFrames
using ACTRMCEModels
using ACTRMCEModels: compute_cf_prob
using ACTRMCEModels: compute_df_prob
using LaTeXStrings
#######################################################################################################################################
#                                                   setup model
#######################################################################################################################################
# specify model parameters: partial matching, noise, mismatch penalty, activation noise
Θ = (mmp = true, noise = true, δ = 0.8, s = 0.20, blc = 0.0, τ = -100)
# base level constant activation for each chunk: F∧B, F̄∧B, F∧B̄, F̄∧B̄
blcs = [-1.0, -1.5, 0, 0.5]
model = ACTRCPI(; blcs, Θ...)
#######################################################################################################################################
#                                                   run example
#######################################################################################################################################
df = DataFrame(judgment = String[], value = Float64[])
# probability of B
p_B = judge(model; B = true)

push!(df, ["p(B)" p_B])

# probability of F ∧ B
p_FB = judge(model; F = true, B = true)
push!(df, ["p(F∧B)" p_FB])

# probability of F
p_F = judge(model; F = true)
push!(df, ["p(F)" p_F])

# probability of F ∧ B
p_ForB = judge_disjunction(model; F = true, B = true)
push!(df, ["p(F∨B)" p_ForB])
df

cf_prob = compute_cf_prob(model, p_FB, p_B)
df_prob = compute_df_prob(model, p_ForB, p_F)
