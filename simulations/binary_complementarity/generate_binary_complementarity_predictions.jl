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
using LaTeXStrings
using Plots
##########################################################################################################
#                                               plot configuration
##########################################################################################################
config = (
    fill = true,
    levels = 20,
    xlabel = L"$n_A$",
    ylabel = L"$n_{\bar{A}}$",
    frame_style = :box,
    xtickfontsize = 5,
    ytickfontsize = 5,
    xguidefontsize = 7,
    yguidefontsize = 7,
    titlefontsize = 7,
    colorbarfontsize = 6,
    clims = (1, 2),
    colorbar = false,
    color = :glasgow
)
#######################################################################################################################################
#                                                           setup simulation
#######################################################################################################################################
Θ = (mmp = true, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
bl = 0.0
n_sub_events1s = 1:10
n_sub_events2s = 1:10
#######################################################################################################################################
#                                                       plot 1
#######################################################################################################################################
make_prediction(model) = judge(model; cause = :cancer) + judge(model; cause = :non_cancer)
δ1 = 0
preds1 = [
    make_prediction(ACTRUnpacking(;
        n_sub_events1 = i,
        n_sub_events2 = j,
        bl,
        Θ...,
        δ = δ1
    )) for i ∈ n_sub_events1s, j ∈ n_sub_events2s
]
plot1 = contour(
    n_sub_events1s,
    n_sub_events2s,
    preds1,
    title = L"\delta = %$δ1";
    config...
)
#######################################################################################################################################
#                                                       plot 2
#######################################################################################################################################
δ2 = 0.5
preds2 = [
    make_prediction(ACTRUnpacking(;
        n_sub_events1 = i,
        n_sub_events2 = j,
        bl,
        Θ...,
        δ = δ2
    )) for i ∈ n_sub_events1s, j ∈ n_sub_events2s
]
plot2 = contour(
    n_sub_events1s,
    n_sub_events2s,
    preds2,
    title = L"\delta = %$δ2";
    config...
)
#######################################################################################################################################
#                                                       plot 3
#######################################################################################################################################
δ3 = 1.0
preds3 = [
    make_prediction(ACTRUnpacking(;
        n_sub_events1 = i,
        n_sub_events2 = j,
        bl,
        Θ...,
        δ = δ3
    )) for i ∈ n_sub_events1s, j ∈ n_sub_events2s
]
plot3 = contour(
    n_sub_events1s,
    n_sub_events2s,
    preds3,
    title = L"\delta = %$δ3";
    config...
)
#######################################################################################################################################
#                                                       plot 4
#######################################################################################################################################
δ4 = 1.5
preds4 = [
    make_prediction(ACTRUnpacking(;
        n_sub_events1 = i,
        n_sub_events2 = j,
        bl,
        Θ...,
        δ = δ4
    )) for i ∈ n_sub_events1s, j ∈ n_sub_events2s
]
plot4 = contour(
    n_sub_events1s,
    n_sub_events2s,
    preds4,
    title = L"\delta = %$δ4";
    config...
)

color_bar = plot(
    plot4;
    framestyle = :none,
    title = "",
    config...,
    colorbar = true,
    xlabel = "",
    ylabel = ""
)

p1 = plot(plot1, plot2, plot3, plot4; layout = (2, 2))
plot(p1, color_bar; dpi = 300, size = (230, 200), layout = @layout [a b{0.05w}])
savefig("actr_binary_complementarity_predictions.png")
