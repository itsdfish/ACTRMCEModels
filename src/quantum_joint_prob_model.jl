"""
    QuantumJointProb{T}

# Fields

- `θ::T`: rotation between bases. θ ∈ [-.5, .5]
- ` ψ₁::T`: probability of first event in basis, e.g., ψ = [√(ψ₁), -√(1 - ψ₁)]
"""
mutable struct QuantumJointProb{T}
    θ::T
    ψ₁::T
end

function QuantumJointProb(; θ, ψ₁)
    return QuantumJointProb(θ, ψ₁)
end

"""
    compute_prob_B(model::QuantumJointProb)

Computes the probability of bank teller (B). 

# Arguments

- `model::QuantumJointProb`: a quantum cognition model of joint probability 
"""
function compute_prob_B(model::QuantumJointProb)
    (; θ, ψ₁) = model
    ψ = [√(ψ₁), -√(1 - ψ₁)]
    U = 𝕌(θ)
    Pb = U * [1 0; 0 0] * U'
    proj_B = Pb * ψ
    return norm(proj_B)^2
end

"""
    compute_prob_FB(model::QuantumJointProb)

Computes the probability of feminist and then bank teller (F → B).    

# Arguments

- `model::QuantumJointProb`: a quantum cognition model of joint probability 
"""
function compute_prob_FB(model::QuantumJointProb)
    (; θ, ψ₁) = model
    ψ = [√(ψ₁), -√(1 - ψ₁)]
    U = 𝕌(θ)
    Pf = [1 0; 0 0]
    Pb = U * [1 0; 0 0] * U'
    proj_FB = Pb * Pf * ψ
    return norm(proj_FB)^2
end

"""
   compute_prob_F(model::QuantumJointProb)

Computes the probability of feminist (B). 

# Arguments

- `model::QuantumJointProb`: a quantum cognition model of joint probability 
"""
function compute_prob_F(model::QuantumJointProb)
    (; ψ₁) = model
    ψ = [√(ψ₁), -√(1 - ψ₁)]
    Pf = [1 0; 0 0]
    proj_F = Pf * ψ
    return norm(proj_F)^2
end

"""
    𝕌(θ)

Computes a unitary (rotation) matrix.

# Arguments

- `θ`: rotation between bases. θ ∈ [-.5, .5]
"""
function 𝕌(θ)
    return [
        cos(π * θ) -sin(π * θ)
        sin(π * θ) cos(π * θ)
    ]
end
