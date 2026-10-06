#######################################################################################################################################
#                                                           load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using ACTRModels
using Distributions
using ACTRMCEModels
using LaTeXStrings
using StatsPlots
using Random
using Turing
using TuringUtilities
include("order_effect_turing_model.jl")
Random.seed!(87544)
#######################################################################################################################################
#                                                           specify data
#######################################################################################################################################
joint_probs = [
    # clinton-gore: (h_c, h_g), (h̄_c, h_g), (h_c, h̄_g), (h̄_c,h̄_g)
    [0.4899, 0.1767, 0.0447, 0.2886],
    # gore-clinton: (h_c, h_g), (h̄_c, h_g), (h_c, h̄_g), (h̄_c,h̄_g)
    [0.5625, 0.1991, 0.0255, 0.2130]
]
_n = [501, 501]
y = map(i -> Int.(round.(joint_probs[i] .* _n[i])), 1:2)
n = sum.(y)
data_order_effect = joint_probs[1] .- joint_probs[2]
#######################################################################################################################################
#                                                           specify model
#######################################################################################################################################
Θ = (sa = true, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)

chunk_counts = (
    n_ch = 5,
    n_ch̄ = 3,
    n_gh = 3,
    n_gh̄ = 1
)

# chunk_counts = (
#     n_ch = 1,
#     n_ch̄ = 3,
#     n_gh = 3,
#     n_gh̄ = 5
# )

turing_model = actr_order_effect(y, n, chunk_counts, Θ)
chains = sample(turing_model, NUTS(1000, 0.65), MCMCThreads(), 1000, 4)
#######################################################################################################################################
#                                                       joint probabilities: Clinton - Gore
#######################################################################################################################################
pyplot()
plot_chunk_counts = (
    norm = true,
    leg = false,
    grid = false,
    color = RGB(119/255, 82/255, 128/255)
)

pred_model = predict_distribution(;
    simulator = p -> rand(ACTROrderEffect(; p..., chunk_counts..., Θ...), n...),
    model = turing_model,
    func = x -> x[1] / sum(x[1])
)

_post_preds = generated_quantities(pred_model, chains)
post_preds_clinton_gore_probs = stack(_post_preds, dims = 1)
reorder_idx = [1, 3, 2, 4]
histogram(
    post_preds_clinton_gore_probs[:, reorder_idx];
    plot_chunk_counts...,
    xlims = (0, 0.6),
    layout = (2, 2)
)
#######################################################################################################################################
#                                                       joint probabilities: Gore - Clinton
#######################################################################################################################################
pred_model = predict_distribution(;
    simulator = p -> rand(ACTROrderEffect(; p..., chunk_counts..., Θ...), n...),
    model = turing_model,
    func = x -> x[2] / sum(x[2])
)

_post_preds = generated_quantities(pred_model, chains)
post_preds_gore_clinton_probs = stack(_post_preds, dims = 1)
histogram(
    post_preds_gore_clinton_probs[:, reorder_idx];
    plot_chunk_counts...,
    xlims = (0, 0.6),
    layout = (2, 2)
)
#######################################################################################################################################
#                                                       order effects:
#######################################################################################################################################
pred_model = predict_distribution(;
    simulator = p -> rand(ACTROrderEffect(; p..., chunk_counts..., Θ...), n...),
    model = turing_model,
    func = compute_order_effects
)

_post_preds = generated_quantities(pred_model, chains)
post_preds_order_effects = stack(_post_preds, dims = 1)

order_effect_plot = histogram(
    post_preds_order_effects[:, reorder_idx];
    xlabel = ["yes,yes" "yes,no" "no,yes" "no,no"],
    plot_chunk_counts...,
    bins = 15,
    xlims = (-0.25, 0.25),
    ylims = (0, 40),
    layout = (2, 2)
)
vline!(data_order_effect[reorder_idx]', color = :black)
# correlation between order effects
cor(post_preds_order_effects[:, 1], post_preds_order_effects[:, 4])
#######################################################################################################################################
#                                                       q value
#######################################################################################################################################
pred_model = predict_distribution(;
    simulator = p -> rand(ACTROrderEffect(; p..., chunk_counts..., Θ...), n...),
    model = turing_model,
    func = compute_q1
)

_post_preds = generated_quantities(pred_model, chains)
post_preds_q_1 = stack(_post_preds, dims = 1)

q1_plot = histogram(
    post_preds_q_1;
    xlabel = L"q",
    plot_chunk_counts...,
    bins = 30,
    xlims = (-0.11, 0.11),
    ylims = (0, 25)
)
vline!([data_order_effect[1] + data_order_effect[4]], color = :black)

layout = @layout [a{0.75h}; b{0.25h}]

plot(
    order_effect_plot,
    q1_plot,
    axis = font(5),
    grid = false,
    leg = false,
    layout = layout,
    size = (230, 250),
    dpi = 300
)
savefig("actr_order_effect_post_pred.eps")
