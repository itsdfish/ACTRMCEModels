@safetestset "quantum conjunction fallacy model" begin
    using ACTRMCEModels
    using Test

    Θ = (
        θ = 0.40,
        ψ₁ = 0.987^2
    )

    model = QuantumJointProb(; Θ...)

    p_B = compute_prob_B(model)
    p_FB = compute_prob_FB(model)
    p_F = compute_prob_F(model)

    # pages 120-121 of Quantum Models of Cognition and Decision (2012)
    @test p_F ≈ 0.97553 atol = 0.005
    @test p_B ≈ 0.024472 atol = 0.005
    @test p_FB ≈ 0.0955 atol = 0.005
end
