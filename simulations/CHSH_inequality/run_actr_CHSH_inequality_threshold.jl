#######################################################################################################################################
#                                                      load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using ACTRModels
using Combinatorics
using DataFrames
using Distributions
using ACTRMCEModels
using ACTRMCEModels: get_chsh_test
using LaTeXStrings
using Plots
using Random
using RectiGrids
include("utilities.jl")
Random.seed!(7954)
#######################################################################################################################################
#                                                           setup model
#######################################################################################################################################
# 16 joint events based on 4 binary events 
joint_events = RectiGrids.grid(
    agreeable = [true, false],
    honest = [true, false],
    intelligent = [true, false],
    unusual = [true, false]
)
# Dirichelet distribution for joint event probabilities
joint_prob_dist = make_dirichlet(0.65, 20)
# inter-arrival time for events  
time_dist = Exponential(100)
# number of events experienced 
n_events = 500
# repetitions of simulation 
n_reps = 1000
# specify model parameters
Θ = (
    bll = true,
    mmp = false,
    noise = true,
    d = 0.50,
    s = 0.20,
    blc = 0.0,
    τ = -0.50
)
# base level constants set to zero for learning 
blcs = fill(0.0, 16)
#######################################################################################################################################
#                                                            run model
#######################################################################################################################################
predictions = run_learning_sim(;
    blcs,
    n_reps,
    n_events,
    joint_events,
    time_dist,
    joint_prob_dist,
    Θ...
)
#######################################################################################################################################
#                                                               plot results
#######################################################################################################################################
# pyplot()
config = (
    norm = true,
    leg = false,
    grid = false,
    color = RGB(75/255, 135/255, 191/255)
)
marginal_config = (
    xlims = (0, 0.25),
    ylims = (0, 55)
)

agreeable_plot =
    histogram(predictions.sd_agreeable; xlabel = "agreeable", marginal_config..., config...)
honest_plot =
    histogram(predictions.sd_honest; xlabel = "honest", marginal_config..., config...)
intelligent_plot = histogram(
    predictions.sd_intelligent;
    xlabel = "intelligent",
    marginal_config...,
    config...
)
unusual_plot =
    histogram(predictions.sd_unusual; xlabel = "unusual", marginal_config..., config...)
marginal_plots =
    plot(agreeable_plot, honest_plot, intelligent_plot, unusual_plot, layout = (2, 2))
chsh_plot = histogram(predictions.chsh; xlabel = "max |CHSH|", ylims = (0, 4), config...)
vline!(chsh_plot, [2.0], color = :black, linewidth = 1.5, linestyle = :dash)
layout = @layout [a{0.75h}; b{0.25h}]

plot(
    marginal_plots,
    chsh_plot,
    axis = font(5),
    grid = false,
    leg = false,
    layout = layout,
    size = (230, 250),
    dpi = 300
)
# savefig("actr_marginal_and_chsh_inequality_predictions.eps")

# # proportion violating CHSH inequality
# mean(predictions.chsh .> 2)
# # proportion violating Tsirelson inequality
# mean(predictions.chsh .> 2 * sqrt(2))
