@safetestset "get_chsh_inequality" begin
    using ACTRMCEModels
    using ACTRMCEModels: get_chsh_test
    using Test

    # From Table 1
    # TT, FT, TF, FF
    probs_AI = [0.271 0.084 0.175 0.469]
    probs_HI = [0.335 0.021 0.035 0.610]
    probs_HU = [0.296 0.088 0.073 0.543]
    probs_AU = [0.115 0.269 0.331 0.285]
    chsh = get_chsh_test(probs_AI, probs_HI, probs_HU, probs_AU)
    @test chsh ≈ 2.25 atol = 0.02
end

@safetestset "compute_marginals" begin
    using ACTRMCEModels
    using ACTRMCEModels: compute_marginals
    using Test

    # TT, FT, TF, FF
    joint_probs = Dict([:A, :B] => [0.20, 0.10, 0.4, 0.3])

    prob_A = compute_marginals(joint_probs, :A)
    @test length(prob_A) == 1
    @test prob_A[1] ≈ 0.60

    prob_B = compute_marginals(joint_probs, :B)
    @test length(prob_B) == 1
    @test prob_B[1] ≈ 0.30
end

@safetestset "make_joint_event_probs" begin
    using ACTRMCEModels
    using ACTRMCEModels: make_joint_event_probs
    using Test

    θ = 0.60
    probs = make_joint_event_probs(θ)
    array = reshape(probs, 2, 2, 2, 2)

    # test marginals 
    @test θ ≈ sum(array, dims = (1, 2, 3))[1]
    @test θ ≈ sum(array, dims = (1, 2, 4))[1]
    @test θ ≈ sum(array, dims = (1, 3, 4))[1]
    @test θ ≈ sum(array, dims = (2, 3, 4))[1]
    @test probs[1] ≈ θ^4
    @test probs[2] ≈ θ^3 * (1 - θ)
    @test probs[16] ≈ (1 - θ)^4
end

@safetestset "make_dirichlet" begin
    using ACTRMCEModels
    using ACTRMCEModels: make_dirichlet
    using Distributions
    using Test

    θ = 0.65
    n = 20
    dist = make_dirichlet(θ, n)
    probs = mean(dist)
    array = reshape(probs, 2, 2, 2, 2)

    # test marginals 
    @test θ ≈ sum(array, dims = (1, 2, 3))[1]
    @test θ ≈ sum(array, dims = (1, 2, 4))[1]
    @test θ ≈ sum(array, dims = (1, 3, 4))[1]
    @test θ ≈ sum(array, dims = (2, 3, 4))[1]
    # test joint 
    @test probs[1] ≈ θ^4
    @test probs[2] ≈ θ^3 * (1 - θ)
    @test probs[16] ≈ (1 - θ)^4

    αs = reshape(dist.alpha, 2, 2, 2, 2)
    @test sum(αs) ≈ n
    # standard deviation of marginal 
    α, β = sum(αs, dims = (1, 2, 3))[:]
    @test std(Beta(α, β)) ≈ 0.104 atol = 1e-3
end

@safetestset "get_ev_prod" begin
    using ACTRMCEModels
    using ACTRMCEModels: get_ev_prod
    using Test

    p = [0.1, 0.3, 0.4, 0.2]
    @test sum(p) ≈ 1
    ev = get_ev_prod(p)
    @test ev ≈ -0.40
end
