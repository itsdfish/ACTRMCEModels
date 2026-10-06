@safetestset "populate_memory" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, noise = true, δ = 0.8, s = 0.20, blc = 0.0, τ = -100)
    blcs = [-1.0, -1.5, 0, 0.5]
    model = ACTRCPI(; blcs, Θ...)
    chunks = model.actr.declarative.memory

    @test length(chunks) == 4

    expected_slots = [
        (F = true, B = true),
        (F = false, B = true),
        (F = true, B = false),
        (F = false, B = false)
    ]
    slots = map(c -> c.slots, chunks)

    @test expected_slots == slots
end

@safetestset "judge 1" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, noise = true, δ = 1, s = 0.20, blc = 0.0, τ = -100)
    blcs = [-1.0, -1.5, 0, 0.5]
    model = ACTRCPI(; blcs, Θ...)

    judge(model; F = true)

    chunks = get_chunks(model.actr; F = true)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 0)

    chunks = get_chunks(model.actr; F = false)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 1)
end

@safetestset "judge 1" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, noise = true, δ = 1, s = 0.20, blc = 0.0, τ = -100)
    blcs = [-1.0, -1.5, 0, 0.5]
    model = ACTRCPI(; blcs, Θ...)

    judge(model; F = true, B = true)

    chunks = get_chunks(model.actr; F = true, B = true)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 0)

    chunks = get_chunks(model.actr; F = true, B = false)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 1)

    chunks = get_chunks(model.actr; F = false, B = true)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 1)

    chunks = get_chunks(model.actr; F = false, B = false)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 2)
end

@safetestset "compute_identities 1" begin
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: compute_identities
    using Test

    for _ ∈ 1:100
        p_FB, p_F̄B, p_FB̄, p_F̄B̄ = rand(Dirichlet(fill(1, 4)))
        p_F = p_FB + p_FB̄
        p_B = p_FB + p_F̄B
        p_ForB = p_F + p_B - p_FB
        z = compute_identities(p_F, p_B, p_FB, p_F̄B, p_FB̄, p_F̄B̄, p_ForB)
        @test all(isapprox.(z, 0; atol = 1e-14))
    end
end

@safetestset "compute_identities 2" begin
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: compute_identities
    using Test

    Θ = (mmp = true, noise = true, δ = 0.0, s = 0.20, blc = 0.0, τ = -100)

    for _ ∈ 1:100
        blcs = rand(Normal(0, 1), 4)
        model = ACTRCPI(; blcs, Θ...)
        z = predict_expected_identities(model)
        @test all(isapprox.(z, 0; atol = 1e-14))
    end
end

@safetestset "learn!" begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using Random
    using Test

    Random.seed!(55325)

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
    # the number of experienced events in a simulation 
    n_events = 10_000
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
    model = ACTRCPI(; blcs, Θ...)

    learn!(model, n_events, event_set, time_dist, event_dist)

    chunks = get_chunks(model.actr; F = true, B = true)
    @test chunks[1].N / n_events ≈ event_probs[1] atol = 1e-2

    chunks = get_chunks(model.actr; F = false, B = true)
    @test chunks[1].N / n_events ≈ event_probs[2] atol = 1e-2

    chunks = get_chunks(model.actr; F = true, B = false)
    @test chunks[1].N / n_events ≈ event_probs[3] atol = 1e-2

    chunks = get_chunks(model.actr; F = false, B = false)
    @test chunks[1].N / n_events ≈ event_probs[4] atol = 1e-2
end

@safetestset "compute_cf_prob" begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: compute_cf_prob
    using Random
    using Test

    Random.seed!(2543)
    n_retrievals = 10
    Θ = (mmp = true, noise = true, δ = 1, s = 0.20, blc = 0.0, τ = -100, n_retrievals)
    blcs = [-1.0, -1.5, 0, 0.5]
    model = ACTRCPI(; blcs, Θ...)

    n_retrievals = 10
    p_B = 0.078
    p_FB = 0.267
    x1 = rand(Binomial(n_retrievals, p_B), 100_000)
    x2 = rand(Binomial(n_retrievals, p_FB), 100_000)
    p = mean(x2 .> x1)

    @test compute_cf_prob(model, p_FB, p_B) ≈ p atol = 1e-2
end

@safetestset "judge_disjunction" begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: judge_disjunction
    using Random
    using Test

    Θ = (mmp = true, noise = true, δ = 1, s = 0.20, blc = 0.0, τ = -100, n_retrievals = 10)

    function sim(Θ)
        blcs = rand(Normal(0, 1), 4)
        δ = rand(Uniform(0, 2))
        model = ACTRCPI(; blcs, Θ..., δ)
        p_B = judge(model; B = true)
        p_F = judge(model; F = true)
        p_ForB = judge_disjunction(model; F = true, B = true)
        return p_ForB < max(p_B, p_F)
    end
    # the expected judgments p(F∨B) ≥ p(F), and therefore cannot exceed 50% rate 
    @test sum(map(_ -> sim(Θ), 1:100_000)) == 0
end

@safetestset "judge_disjunction with threshold" begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: judge_disjunction
    using Random
    using Test

    Θ = (mmp = true, noise = true, δ = 1, s = 0.20, blc = 0.0, n_retrievals = 10)

    function sim(Θ)
        blcs = rand(Normal(0, 1), 4)
        δ = rand(Uniform(0, 2))
        τ = rand(Normal(0, 0.50))
        model = ACTRCPI(; blcs, Θ..., δ, τ)
        p_B = judge(model; B = true)
        p_F = judge(model; F = true)
        p_ForB = judge_disjunction(model; F = true, B = true)
        return p_ForB < max(p_B, p_F)
    end
    # the expected judgments p(F∨B) < p(F) is possible
    @test sum(map(_ -> sim(Θ), 1:100_000)) > 0
end

@safetestset "conjunction fallacy rate threshold" begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: compute_cf_prob
    using Random
    using Test

    Θ = (mmp = false, noise = true, s = 0.20, blc = 0.0, τ = 0, n_retrievals = 10)

    function sim(Θ)
        blcs = rand(Normal(0, 1), 4)
        τ = rand(Normal(0, 0.5))
        model = ACTRCPI(; blcs, Θ..., τ)
        p_B = judge(model; B = true)
        p_FB = judge(model; F = true, B = true)
        return compute_cf_prob(model, p_FB, p_B)
    end

    cf_probs = map(_ -> sim(Θ), 1:10_000)
    @test maximum(cf_probs) ≤ 0.50
end

@safetestset "averaging disjunction " begin
    using ACTRModels
    using Distributions
    using ACTRMCEModels
    using ACTRMCEModels: compute_cf_prob
    using Random
    using Test

    Θ = (mmp = false, noise = true, s = 0.20, blc = 0.0, τ = 0, n_retrievals = 10)
    blcs = [0.3, 0.4, -0.3, 0]
    δ = 0.50
    model = ACTRCPIAveraging(; blcs, Θ..., δ)
    p_F = judge(model, F = true)
    p_B = judge(model, B = true)
    p_ForB = judge_disjunction(model, F = true, B = true)
    @test 0.5 * (p_F + p_B) ≈ p_ForB
end
