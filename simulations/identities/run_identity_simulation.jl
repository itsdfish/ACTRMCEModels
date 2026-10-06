#######################################################################################################################################
#                                                   load packages
#######################################################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using ACTRModels
using DataFrames
using Distributions
using ACTRMCEModels
using LaTeXStrings
using StatsPlots
using Random
Random.seed!(9336)
#######################################################################################################################################
#                                                   configure model
#######################################################################################################################################
# fixed ACT-R parameters
Θ = (
    mmp = true,
    noise = true,
    s = 0.20,
    blc = 0.0,
    n_retrievals = 10
)

# number of subjects
n_subj = 100
# repetitions of simulation 
n_reps = 1000
# model type: ACTRCPI or ACTRCPIAveraging
model_type = ACTRCPI
blc_dist = Normal(0, 1)
δs = [0.30, 0.50, 0.80]
τ_dist = -100
δ_dists = map(δ -> truncated(Normal(δ, 0.30), 0, Inf), δs)
#######################################################################################################################################
#                                                   simulate model
#######################################################################################################################################
preds = map(
    δ_dist ->
        simulate_group(
            model_type,
            n_subj,
            n_reps;
            func = predict_noisy_identities,
            Θ,
            blc_dist,
            δ_dist,
            τ_dist
        ),
    δ_dists
)

sds = map(x -> std(x, dims = 1)[:], preds)
sds5 = map(x -> std(x[1:5, :], dims = 1)[:], preds)
mean.(sds5)

preds_equal_z = simulate_group(
    model_type,
    n_subj,
    n_reps;
    func = predict_noisy_identities,
    Θ,
    blc_dist,
    δ_dist = 0.0,
    τ_dist
)

sds_equal_z = std(preds_equal_z, dims = 1)[:]
#######################################################################################################################################
#                                                   plot results
#######################################################################################################################################
pyplot()

config = (
    xlims = (-0.01, 0.17),
    axis = font(7),
    legendfontsize = 4,
    legendtitle = L"\mu_{\delta}",
    legendtitlefontsize = 5,
    xlabel = "sd",
    ylabel = "Density",
    label = δs',
    alpha = 0.7,
    color = [
        RGB(45/255, 63/255, 166/255),
        RGB(138/255, 48/255, 54/255),
        RGB(37/255, 138/255, 57/255)
    ]',
    grid = false,
    norm = true
)
sd_plot = histogram(sds; config...)
histogram!(sds_equal_z, label = L"\delta_i = 0")
labels = [L"z_1", L"z_2", L"z_3", L"z_4", L"z_5/2", L"z_6/2"]
violin_plot = violin(
    preds[2]',
    xticks = (1:length(labels), labels),
    ylabel = "x",
    ylims = (0, 0.60),
    leg = false,
    grid = false,
    color = RGB(138/255, 48/255, 54/255),
    alpha = 0.70,
    axis = font(7)
)
hline!([mean(preds[2])], color = :black, linestyle = :dash)

plot(sd_plot, violin_plot, layout = (2, 1), size = (230, 200), dpi = 300)

savefig("actr_identity_predictions.png")
