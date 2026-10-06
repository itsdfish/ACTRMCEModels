cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using DataFrames
using Distributions
using ExtendingNewellsTest
using LaTeXStrings
using Plots

Θ = (
    θ = 0.3,
    ψ₁ = 0.90
)
model = QuantumJointProb(; Θ...)

df = DataFrame(judgment = String[], value = Float64[])
# probability of B
p_B = compute_prob_B(model)

push!(df, ["p(B)" p_B])

# probability of F ∧ B
p_FB = compute_prob_FB(model)
push!(df, ["p(F∧B)" p_FB])

# probability of F
p_F = compute_prob_F(model)
push!(df, ["p(F)" p_F])
