##########################################################################################################
#                                               load dependencies
##########################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using CSV
using DataFrames
using LaTeXStrings
using PriorityHeuristicModels
using PriorityHeuristicModels: to_decision_index
using Plots
##########################################################################################################
#                                               plot configuration
##########################################################################################################
config = (
    fill = true,
    levels = 2,
    xlabel = L"$x_G$",
    ylabel = L"$x_L$",
    xlims = (0, 250),
    ylims = (-250, 0),
    frame_style = :box,
    colorbar_ticks = (1:3, ["safe", "risky", "next feature"]),
    xaxis = font(6),
    yaxis = font(6),
    titlefontsize = 7,
    colorbarfontsize = 5,
    color = [:black, RGB(83/255, 128/255, 82/255), :yellow],
    colorbar = false
)
##########################################################################################################
#                                              load data
##########################################################################################################
df = CSV.read("../../Barkan_2003_data/Barkan_2003_data.csv", DataFrame)
gambles = unique(df, :trial_id)
# identify gambles in which a loss occurred in stage 1
gambles.is_stage1_loss .=
    (
        gambles.lose_plan_take_final_take +
        gambles.lose_plan_take_final_reject +
        gambles.lose_plan_reject_final_take +
        gambles.lose_plan_reject_final_reject
    ) .> 0
# stage 1 losses
gambles_stage1_loss = filter(x -> x.is_stage1_loss, eachrow(gambles))
# stage 1 wins
gambles_stage1_win = filter(x -> !x.is_stage1_loss, eachrow(gambles))
##########################################################################################################
#                            generate predictions for decision
##########################################################################################################
model = PriorityHeuristic()
x_Gs = range(0, 250, length = 500)
x_Ls = range(0, -250, length = 500)
p_outcome = 0.50
# final decisions for loss in first stage
decisions_loss = [decide(model, x_L, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_loss = map(x -> to_decision_index(x), decisions_loss)
decision_idxs_loss[1, 2] = 3

# final decisions for gain in first stage
decisions_gain = [decide(model, x_G, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_idxs_gain = map(x -> to_decision_index(x), decisions_gain)
decision_idxs_gain[1, 2] = 3

# planned decisions
decisions_planned = [decide(model, 0, x_G, x_L, p_outcome) for x_L ∈ x_Ls, x_G ∈ x_Gs]
decision_planned_idxs = map(x -> to_decision_index(x), decisions_planned)
# needed to properly display color bar
decision_planned_idxs[1, 1] = 1
decision_planned_idxs[1, 2] = 3
##########################################################################################################
#                            generate plots for  decisions
##########################################################################################################
pyplot()
final_loss_plot =
    contour(x_Gs, x_Ls, decision_idxs_loss, title = "final decision, loss"; config...)

final_gain_plot = contour(
    x_Gs,
    x_Ls,
    decision_idxs_gain,
    title = "final decision, gain";
    config...,
    xticks = nothing,
    xlabel = ""
)

planned_plot = contour(
    x_Gs,
    x_Ls,
    decision_planned_idxs,
    title = "planned decision";
    config...,
    xticks = nothing,
    xlabel = ""
)
##########################################################################################################
#                                           add outcomes 
##########################################################################################################
scatter!(
    final_gain_plot,
    gambles_stage1_win.gain,
    gambles_stage1_win.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)

scatter!(
    final_loss_plot,
    gambles_stage1_loss.gain,
    gambles_stage1_loss.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)

scatter!(
    planned_plot,
    gambles.gain,
    gambles.loss,
    leg = false,
    color = :darkorange,
    markersize = 1,
    markerstrokewidth = 0.0
)
##########################################################################################################
#                                           combine plots 
##########################################################################################################
plot(planned_plot, final_gain_plot, final_loss_plot, layout = (3, 1), size = (240, 240))
savefig("decision_predictions.eps")
