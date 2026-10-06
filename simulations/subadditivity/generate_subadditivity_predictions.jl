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
using ACTRMCEModels: simulate_trial
#######################################################################################################################################
#                                                           setup simulation
#######################################################################################################################################
Θ = (mmp = true, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
bl = 0.0
max_sub_events = 5
δs = range(0, 1.5, length = 5)
unpacking_factors = Vector{Vector{Float64}}()
#######################################################################################################################################
#                                                           run simulation
#######################################################################################################################################
for δ ∈ δs
    temp = zeros(max_sub_events)
    for i ∈ 1:max_sub_events
        model = ACTRUnpacking(; n_sub_events1 = i, bl, Θ..., δ)
        temp[i] = simulate_trial(model)
    end
    push!(unpacking_factors, temp)
end
#######################################################################################################################################
#                                                           plot simulation
#######################################################################################################################################
pyplot()
plot(
    1:max_sub_events,
    unpacking_factors,
    ylims = (0, 5),
    xlabel = "Number of Sub-events",
    ylabel = "Unpacking Factor",
    line_z = δs',
    color = cgrad([RGB(138/255, 141/255, 227/255), RGB(10/255, 12/255, 66/255)]),
    legendtitle = "δ",
    colorbar = false,
    grid = false,
    label = δs',
    leg = :topleft,
    axis = font(6),
    legendtitlefontsize = 5,
    legendfontsize = 4,
    size = (230, 150)
)
savefig("actr_subadditivity.eps")
