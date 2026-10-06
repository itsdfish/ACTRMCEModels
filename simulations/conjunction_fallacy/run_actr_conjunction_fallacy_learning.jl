#######################################################################################################################################
#                                                   load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using ACTRModels
using DataFrames
using Distributions
using ACTRMCEModels
using LaTeXStrings
using Plots
using Random
Random.seed!(62510)
include("learning_simulation_utilities.jl")
#######################################################################################################################################
#                                                   setup simulation
#######################################################################################################################################
# specify model parameters: partial matching, noise, mismatch penalty, activation noise
Θ = (
    mmp = true,
    bll = true,
    noise = true,
    δ = 0.8,
    d = 0.5,
    s = 0.20,
    blc = 0.0,
    τ = -100
)
# base level constant activation for each chunk: F∧B, F̅∧B, F∧B̅, F̅∧B̅
blcs = fill(0.0, 4)
# the number of repetitions of the simulation
n_reps = 1000
# the number of experienced events in a simulation 
n_events = 500
# inter-arrival time between events 
time_dist = Exponential(100)
# joint probability distribution over events 
event_probs = [0.2, 0.1, 0.3, 0.4]
# event distribution
event_dist = Categorical(event_probs)
# the events 
event_set = [
    (F = true, B = true),
    (F = false, B = true),
    (F = true, B = false),
    (F = false, B = false)
]
#######################################################################################################################################
#                                                   run simulation
#######################################################################################################################################
results1 =
    run_learning_sim(;
        blcs,
        n_reps,
        n_events,
        event_set,
        time_dist,
        event_dist,
        Θ...,
        δ = 0.8
    )

percent_cf = mean(results1[:, 1] .< results1[:, 2])

results2 = run_learning_sim(;
    blcs,
    n_reps,
    n_events,
    event_set,
    time_dist,
    event_dist,
    Θ...,
    δ = 0.3
)

percent_cf = mean(results2[:, 1] .< results2[:, 2])
#######################################################################################################################################
#                                                   plot results 
#######################################################################################################################################
pyplot()
scatter(
    results1[:, 1],
    results1[:, 2];
    xlabel = "p(F)",
    ylabel = "p(F∧B)",
    axis = font(7),
    lims = (0, 1),
    size = (230, 170),
    markersize = 2,
    grid = false,
    markerstrokewidth = 0.5,
    color = RGB(119/255, 82/255, 128/255),
    leg = false,
    alpha = 0.80,
    framestyle = :box,
    dpi = 300
)

scatter!(
    results2[:, 1],
    results2[:, 2];
    xlabel = "p(B)",
    ylabel = "p(F∧B)",
    axis = font(7),
    lims = (0, 1),
    size = (230, 170),
    markersize = 2,
    markerstrokewidth = 0.5,
    color = RGB(83/255, 128/255, 82/255),
    leg = false,
    alpha = 0.80
)

annotate!([(
    event_probs[1] + event_probs[2],
    event_probs[1],
    text("X", 6, :center, :black)
)])

plot!(0:0.1:1, 0:0.1:1, color = :black, linestyle = :solid, linewidth = 0.5)

savefig("actr_cf_learning_scatter.png")
