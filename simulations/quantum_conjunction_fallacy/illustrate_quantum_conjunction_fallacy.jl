cd(@__DIR__)
using Pkg
Pkg.activate("../..")
using Revise
using Distributions
using ExtendingNewellsTest
using ExtendingNewellsTest: 𝕌
using LaTeXStrings
using Plots

θ = 0.3
ψ₁ = 0.90

model = QuantumJointProb(; θ, ψ₁)
ψ = [√(ψ₁), -√(1 - ψ₁)]
F = [1, 0]
F̄ = [0, 1]

U = 𝕌(θ)
B = U * F
B̄ = U * F̄

Pf = [1 0; 0 0]
Pb = U * [1 0; 0 0] * U'

proj_F = Pf * ψ
proj_FB = Pb * Pf * ψ
proj_B = Pb * ψ

function label_loc(X, θ)
    d = 𝕌(θ) * 0.8
    return X .- d * X * 0.1
end

reflect(X) = X[1] < 0 ? (-1, -1) : (1, 1)

pyplot()

plot(
    [0, F[1]],
    [0, F[2]],
    framestyle = :origin,
    arrow = true,
    xaxis = font(6),
    yaxis = font(6),
    xlims = (-1, 1),
    ylims = (-1, 1),
    grid = false,
    leg = false,
    color = :black,
    linewidth = 1,
    size = (230, 200)
)

plot!(
    [0, F̄[1]],
    [0, F̄[2]],
    framestyle = :origin,
    arrow = true,
    color = :black,
    linewidth = 1
)

plot!(
    [0, B[1]],
    [0, B[2]],
    framestyle = :origin,
    arrow = true,
    color = :black,
    linestyle = :dash,
    linewidth = 1
)

plot!(
    [0, B̄[1]],
    [0, B̄[2]],
    framestyle = :origin,
    arrow = true,
    linestyle = :dash,
    color = :black,
    linewidth = 1
)
# parallel translation of projection to reduce visual clutter
d = -0.02
Δ = d * √(2) / 2
plot!(
    [Δ, proj_B[1] + Δ],
    [-Δ, proj_B[2] - Δ],
    framestyle = :origin,
    color = RGB(83/255, 128/255, 82/255),
    linewidth = 1.6
)

plot!(
    [0, ψ[1]],
    [0, ψ[2]],
    framestyle = :origin,
    arrow = true,
    color = :black,
    linewidth = 1,
    linestyle = :solid
)

plot!(
    [ψ[1], ψ[1]],
    [ψ[2], F[2]],
    framestyle = :origin,
    arrow = true,
    color = RGB(119/255, 82/255, 128/255),
    linewidth = 1,
    linestyle = :dash
)

cx, cy = reflect(B)
plot!(
    [F[1], cx * proj_FB[1]],
    [F[2], cy * proj_FB[2]],
    framestyle = :origin,
    arrow = true,
    color = RGB(119/255, 82/255, 128/255),
    linewidth = 1,
    linestyle = :dash
)

plot!(
    [0, proj_FB[1]],
    [0, proj_FB[2]],
    framestyle = :origin,
    color = RGB(119/255, 82/255, 128/255),
    linewidth = 1.6
)

Δf = 0.02
plot!(
    [Δf, proj_F[1] + Δf],
    [Δf, proj_F[2] + Δf],
    framestyle = :origin,
    color = RGB(119/255, 82/255, 128/255),
    linewidth = 1.4
)

cx, cy = reflect(B)
plot!(
    [ψ[1], cx * proj_B[1]],
    [ψ[2], cy * proj_B[2]],
    framestyle = :origin,
    arrow = true,
    color = RGB(83/255, 128/255, 82/255),
    linewidth = 1,
    linestyle = :dash
)

ψ_label = label_loc(ψ, θ)
annotate!(ψ_label[1], ψ_label[2], text(L"\psi", :black, 8))

F_label = label_loc(F, -θ)
annotate!(F_label[1], F_label[2], text(L"F", :black, 8))

F̅_label = label_loc(F̄, θ)
annotate!(F̅_label[1], F̅_label[2], text(L"\bar{F}", :black, 8))

B_label = label_loc(B, -θ)
annotate!(B_label[1], B_label[2], text(L"B", :black, 8))

B̅_label = label_loc(B̄, θ)
annotate!(B̅_label[1], B̅_label[2], text(L"\bar{B}", :black, 8))
savefig("quantum_conjunction_fallacy_demo.eps")
# let
# 	@test norm(S) ≈ 1
# 	@test norm(Z) ≈ 1 atol = 1e-3
# 	@test norm((FB * FB' + F̅B * F̅B') * S)^2 ≈ S[1]^2 + S[3]^2
# 	@test B' * B̅ ≈ 0 atol = 1e-6
# 	@test norm(_S) ≈ 1

# 	θ = 0.4
# 	@test compute_B(Z, F, θ) ≈ 0.024472 atol = 0.002
# 	@test compute_FB(Z, F, θ) ≈ 0.093155 atol = 0.002
# end
