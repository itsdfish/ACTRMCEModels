@safetestset "populate_memory" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, δ = 1, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 3, n_sub_events2 = 4, bl, Θ...)

    chunks = model.actr.declarative.memory

    @test length(chunks) == 7

    cancer_chunks = get_chunks(model.actr; cause = :cancer)
    @test length(cancer_chunks) == 3
    cancer_types = map(c -> c.slots.type, cancer_chunks)
    @test cancer_types == [1, 2, 3]

    noncancer_chunks = get_chunks(model.actr; cause = :non_cancer)
    @test length(noncancer_chunks) == 4
    noncancer_types = map(c -> c.slots.type, noncancer_chunks)
    @test noncancer_types == [4:7;]
end

@safetestset "judge1" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, δ = 1, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 3, n_sub_events2 = 4, bl, Θ...)
    judge(model; cause = :cancer)

    cancer_chunks = get_chunks(model.actr; cause = :cancer)
    act_pm = map(c -> c.act_pm, cancer_chunks)
    @test all(act_pm .≈ 0)

    noncancer_chunks = get_chunks(model.actr; cause = :non_cancer)
    act_pm = map(c -> c.act_pm, noncancer_chunks)
    @test all(act_pm .≈ 1)
end

@safetestset "judge2" begin
    using ACTRModels
    using ACTRMCEModels
    using Test

    Θ = (mmp = true, δ = 1, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 3, n_sub_events2 = 4, bl, Θ...)
    judge(model; cause = :cancer, type = 1)

    cancer_chunks = get_chunks(model.actr; cause = :cancer, type = 1)
    act_pm = map(c -> c.act_pm, cancer_chunks)
    @test all(act_pm .≈ 0)

    chunks = get_chunks(model.actr,==,≠; cause = :cancer, type = 1)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 1)

    chunks = get_chunks(model.actr,≠,≠; cause = :cancer, type = 1)
    act_pm = map(c -> c.act_pm, chunks)
    @test all(act_pm .≈ 2)
end

@safetestset "simulate_trial 1" begin
    using ACTRModels
    using ACTRMCEModels
    using ACTRMCEModels: simulate_trial
    using Test

    Θ = (mmp = true, δ = 1, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 5, bl, Θ...)
    uf = simulate_trial(model)
    @test uf > 1
end

@safetestset "simulate_trial 2" begin
    using ACTRModels
    using ACTRMCEModels
    using ACTRMCEModels: simulate_trial
    using Test

    Θ = (mmp = true, δ = 1000000, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 5, bl, Θ...)
    uf = simulate_trial(model)
    @test uf ≈ 5
end

@safetestset "simulate_trial 3" begin
    using ACTRModels
    using ACTRMCEModels
    using ACTRMCEModels: simulate_trial
    using Test

    Θ = (mmp = true, δ = 0, noise = true, s = 0.20, blc = 0.0, τ = -1000.0)
    bl = 0.0
    model = ACTRUnpacking(; n_sub_events1 = 5, bl, Θ...)
    uf = simulate_trial(model)
    @test uf ≈ 1
end
