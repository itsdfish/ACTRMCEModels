@safetestset "no order effects if no spreading activation" begin
    using Distributions
    using ACTRMCEModels
    using Test

    n_sim = 10_000
    Θ = (sa = false, noise = true, s = 0.20, blc = 0.0, τ = -1000)

    lb = 0
    ub = 6
    μ = 0
    σ = 0
    config1 = (
        n_ch = DiscreteUniform(lb, ub),
        n_ch̄ = DiscreteUniform(lb, ub),
        n_gh = DiscreteUniform(lb, ub),
        n_gh̄ = DiscreteUniform(lb, ub),
        bl_ch = Normal(μ, σ),
        bl_ch̄ = Normal(μ, σ),
        bl_gh = Normal(μ, σ),
        bl_gh̄ = Normal(μ, σ)
    )

    for _ ∈ 1:n_sim
        model = ACTROrderEffect(; config1..., Θ..., γ = 0.0)
        order_effects = compute_order_effects(model)
        @test all(order_effects .≈ 0)
    end
end

@safetestset "no order effects if balanced ratios" begin
    using Distributions
    using ACTRMCEModels
    using Test

    n_sim = 100
    Θ = (sa = true, noise = true, s = 0.20, blc = 0.0, τ = -1000)

    lb = 0
    ub = 6
    μ = 0
    σ = 0

    for _ ∈ 1:n_sim
        n_ch = rand(DiscreteUniform(lb, ub))
        x = rand(DiscreteUniform(lb, ub))
        config1 = (
            n_ch,
            n_ch̄ = n_ch,
            n_gh = x * n_ch,
            n_gh̄ = x * n_ch,
            bl_ch = 0,
            bl_ch̄ = 0,
            bl_gh = 0,
            bl_gh̄ = 0,
            γ = Uniform(2, 5)
        )
        model = ACTROrderEffect(; config1..., Θ...)
        order_effects = compute_order_effects(model)
        @test var(order_effects) ≈ 0 atol = 1e-12
    end
end

@safetestset "no order effects for yes-yes, no-no if equal ratios" begin
    using Distributions
    using ACTRMCEModels
    using Test

    n_sim = 100
    Θ = (sa = true, noise = true, s = 0.20, blc = 0.0, τ = -1000)

    lb = 0
    ub = 6
    μ = 0
    σ = 0

    for _ ∈ 1:n_sim
        n_ch = rand(DiscreteUniform(lb, ub))
        n_ch̄ = rand(DiscreteUniform(lb, ub))
        x = rand(DiscreteUniform(lb, ub))
        config1 = (
            n_ch,
            n_ch̄,
            n_gh = x * n_ch,
            n_gh̄ = x * n_ch̄,
            bl_ch = 0,
            bl_ch̄ = 0,
            bl_gh = 0,
            bl_gh̄ = 0,
            γ = Uniform(2, 5)
        )
        model = ACTROrderEffect(; config1..., Θ...)
        order_effects = compute_order_effects(model)
        @test order_effects[1] ≈ 0 atol = 1e-12
        @test order_effects[4] ≈ 0 atol = 1e-12
        @test order_effects[2] ≈ -order_effects[3] atol = 1e-12
    end
end

@safetestset "no order effects for yes-no, no-yes if ratios are complementary" begin
    using Distributions
    using ACTRMCEModels
    using Test

    n_sim = 100
    Θ = (sa = true, noise = true, s = 0.20, blc = 0.0, τ = -1000)

    lb = 0
    ub = 6
    μ = 0
    σ = 0

    for _ ∈ 1:n_sim
        n_ch = rand(DiscreteUniform(lb, ub))
        n_ch̄ = rand(DiscreteUniform(lb, ub))
        config1 = (
            n_ch,
            n_ch̄,
            n_gh = n_ch̄,
            n_gh̄ = n_ch,
            bl_ch = 0,
            bl_ch̄ = 0,
            bl_gh = 0,
            bl_gh̄ = 0,
            γ = Uniform(2, 5)
        )
        model = ACTROrderEffect(; config1..., Θ...)
        order_effects = compute_order_effects(model)
        @test order_effects[2] ≈ 0 atol = 1e-12
        @test order_effects[3] ≈ 0 atol = 1e-12
        @test order_effects[1] ≈ -order_effects[4] atol = 1e-12
    end
end
