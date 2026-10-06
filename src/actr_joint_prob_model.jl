abstract type AbstractACTRCPI end
"""
    ACTRCPI{A, T}

An ACT-R model of joint probability judgment 

# Fields 

- `actr::A`: an ACT-R model object 
- `blcs::Vector{T}`: base level constants corresponding to chunks for events F∧B, F̄∧B, F∧B̄, F̄∧B̄

# Constructors

    ACTRCPI(; n_retrievals = 10, blcs, Θ...)

where Θ... is an optional set of keyword arguments for ACT-R parameters

    ACTRCPI(actr, blcs, n_retrievals)
"""
mutable struct ACTRCPI{A, T} <: AbstractACTRCPI
    actr::A
    blcs::Vector{T}
    n_retrievals::Int
end

function ACTRCPI(; n_retrievals = 10, blcs, Θ...)
    # create chunks of declarative knowledge
    chunks = populate_memory(blcs)
    # initialize declarative memory
    declarative = Declarative(memory = chunks)
    # create an ACT-R object with activation noise and partial matching
    actr = ACTR(; declarative, Θ...)
    return ACTRCPI(actr, blcs, n_retrievals)
end

"""
    ACTRCPIAveraging{A, T}

An ACT-R model of joint probability judgment which uses averaging for disjunctive probability estimates. 

# Fields 

- `actr::A`: an ACT-R model object 
- `blcs::Vector{T}`: base level constants corresponding to chunks for events F∧B, F̄∧B, F∧B̄, F̄∧B̄

# Constructors

    ACTRCPIAveraging(; n_retrievals = 10, blcs, Θ...)

where Θ... is an optional set of keyword arguments for ACT-R parameters

    ACTRCPIAveraging(actr, blcs, n_retrievals)
"""
mutable struct ACTRCPIAveraging{A, T} <: AbstractACTRCPI
    actr::A
    blcs::Vector{T}
    n_retrievals::Int
end

function ACTRCPIAveraging(; n_retrievals = 10, blcs, Θ...)
    # create chunks of declarative knowledge
    chunks = populate_memory(blcs)
    # initialize declarative memory
    declarative = Declarative(memory = chunks)
    # create an ACT-R object with activation noise and partial matching
    actr = ACTR(; declarative, Θ...)
    return ACTRCPIAveraging(actr, blcs, n_retrievals)
end

"""
    populate_memory(blcs)

Generates chunks for the binary events F and Bool

# Arguments 

- `blcs::Vector{T}`: base level constants corresponding to chunks for events F∧B, F̄∧B, F∧B̄, F̄∧B̄
"""
function populate_memory(blcs)
    chunks = [
        Chunk(; F = true, B = true, bl = blcs[1]),
        Chunk(; F = false, B = true, bl = blcs[2]),
        Chunk(; F = true, B = false, bl = blcs[3]),
        Chunk(; F = false, B = false, bl = blcs[4])
    ]
    return chunks
end

"""
    learn!(model::AbstractACTRCPI, n_events, event_set, time_dist, event_dist)

The model learns the distribution of events in a simulated environment.

# Arguments

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 
- `n_events`: the number of events from which the model learns about the environment 
- `event_set`: a vector of slot-value pairs representing the properities of experienced events 
- `time_dist`: an inter-event time distribution 
- `event_dist`: a probability distribution over `event_set` 
"""
function learn!(model::AbstractACTRCPI, n_events, event_set, time_dist, event_dist)
    for _ ∈ 1:n_events
        idx = rand(event_dist)
        event = event_set[idx]
        tΔ = rand(time_dist)
        increment_time!(model, tΔ)
        add_chunk!(model.actr; event...)
    end
    return nothing
end

"""
    increment_time!(model, tΔ)

# Arguments

- `model`: an ACT-R model 
- `tΔ`: the amount by which time is incremented 
"""
function increment_time!(model, tΔ)
    model.actr.scheduler.time += tΔ
    return nothing
end

"""
    judge(model::AbstractACTRCPI; request...)

Judges the probability of events identified in `request...`.
`request...` serves as a filter for the numerator and as a retrieval request 

# Arguments 

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 
- `request...`: optional keyword arguments serving as a retrieval request
"""
function judge(model::AbstractACTRCPI; request...)
    target = get_chunks(model.actr; request...)
    p, pf = retrieval_prob(model.actr, target; request...)
    return p + 0.50 * pf
end

"""
    judge_disjunction(model::AbstractACTRCPI; request...)

Judges the probability of F or B 

# Arguments 

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 
- `request...`: optional keyword arguments serving as a retrieval request
"""
function judge_disjunction(model::ACTRCPI; request...)
    target = get_chunks(model.actr; F = true)
    push!(target, get_chunks(model.actr; B = true)...)
    target = unique(target)
    p, pf = retrieval_prob(model.actr, target; request...)
    θ = p + 0.50 * pf
    return θ
end

"""
    judge_disjunction(model::ACTRCPIAveraging; request...)

Judges the probability of F or B as the average of P(F) and P(B)

# Arguments 

- `model::ACTRCPIAveraging`: an ACT-R model of joint probability judgment 
- `request...`: optional keyword arguments serving as a retrieval request
"""
function judge_disjunction(model::ACTRCPIAveraging; request...)
    p_F = judge(model; F = request[:F])
    p_B = judge(model; B = request[:B])
    θ = 0.5 * (p_F + p_B)
    return θ
end

function respond(model::AbstractACTRCPI; request...)
    (; n_retrievals) = model
    θ = judge(model; request...)
    return rand(Binomial(n_retrievals, θ)) / n_retrievals
end

function respond_disjunction(model::AbstractACTRCPI; request...)
    (; n_retrievals) = model
    θ = judge_disjunction(model; request...)
    return rand(Binomial(n_retrievals, θ)) / n_retrievals
end

"""
    compute_cf_prob(model::AbstractACTRCPI, p_FB, p_B)

Computes the probability of a conjunction fallacy based on judgments following a binomial distribution 

# Arguments

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 
- `p_FB`: expected judgment for feminist and bank teller 
- `p_B`: expected judgment for bank teller 
"""
function compute_cf_prob(model::AbstractACTRCPI, p_FB, p_B)
    (; n_retrievals) = model
    p = 0.0
    for i ∈ 0:n_retrievals
        p +=
            pdf(Binomial(n_retrievals, p_B), i) * (1 - cdf(Binomial(n_retrievals, p_FB), i))
    end
    return p
end

"""
    compute_df_prob(model::AbstractACTRCPI, p_FB, p_B)

Computes the probability of a disjunction fallacy based on judgments following a binomial distribution. 

# Arguments

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 
- `p_ForB`: expected judgment for feminist or bank teller 
- `p_F`: expected judgment for feminist
"""
function compute_df_prob(model::AbstractACTRCPI, p_ForB, p_F)
    (; n_retrievals) = model
    p = 0.0
    for i ∈ 0:n_retrievals
        p +=
            pdf(Binomial(n_retrievals, p_F), i) * cdf(Binomial(n_retrievals, p_ForB), i)
    end
    return p
end

"""
    predict_expected_identities(model::AbstractACTRCPI)

Generates predictions of the 6 identities investigated in Costello, Watts & Fisher (2018) based on expected judgments.

# Arguments 

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 

# Returns

- `judgments::Vector{Float64}`: a vector of probability judgments for the following identities based on expected judgments.

 1. p(F) + p(F̄ ∧ B) - p(F ∨ B)
 2. p(B) + p(F ∧ B̄) - p(F ∨ B)
 3. p(F ∧ B) + p(F ∧ B̄) - p(F)
 4. p(F ∧ B) + p(F̄ ∧ B) - p(B)
 5. p(F ∧ B̄) + p(F̄ ∧ B) + p(F ∧ B) - p(F ∨ B)
 6. p(F ∧ B̄) + p(F̄ ∧ B) + 2p(F ∧ B) - p(F) - p(B)
 7. p(F) + p(B) - p(F ∧ B) - p(F ∨ B)
"""
function predict_expected_identities(model::AbstractACTRCPI)
    # probability of F
    p_F = judge(model; F = true)

    # probability of B
    p_B = judge(model; B = true)

    # probability of F∧B
    p_FB = judge(model; F = true, B = true)

    # probability of F∧B̄
    p_FB̄ = judge(model; F = true, B = false)

    # probability of F̄∧B
    p_F̄B = judge(model; F = false, B = true)

    # probability of F̄∧B̄
    p_F̄B̄ = judge(model; F = false, B = false)

    # probability of F∨B
    p_ForB = judge_disjunction(model; F = true, B = true)

    z = compute_identities(p_F, p_B, p_FB, p_F̄B, p_FB̄, p_F̄B̄, p_ForB)
    return z
end

"""
    predict_noisy_identities(model::AbstractACTRCPI)

Generates predictions of the 6 identities investigated in Costello, Watts & Fisher (2018) based on noisy judgments.

# Arguments 

- `model::AbstractACTRCPI`: an ACT-R model of joint probability judgment 

# Returns

- `judgments::Vector{Float64}`: a vector of probability judgments for the following identities

 1. p(F) + p(F̄ ∧ B) - p(F ∨ B)
 2. p(B) + p(F ∧ B̄) - p(F ∨ B)
 3. p(F ∧ B) + p(F ∧ B̄) - p(F)
 4. p(F ∧ B) + p(F̄ ∧ B) - p(B)
 5. p(F ∧ B) + p(F̄ ∧ B) + p(F ∧ B̄) - p(F ∨ B)
 6. p(F ∧ B̄) + p(F̄ ∧ B) + p(F̄ ∧ B) + p(F̄ ∧ B̄) - 1
"""
function predict_noisy_identities(model::AbstractACTRCPI)
    # probability of F
    p_F = respond(model; F = true)

    # probability of B
    p_B = respond(model; B = true)

    # probability of F∧B
    p_FB = respond(model; F = true, B = true)

    # probability of F∧B̄
    p_FB̄ = respond(model; F = true, B = false)

    # probability of F̄∧B̄
    p_F̄B̄ = respond(model; F = false, B = false)

    # probability of F̄∧B
    p_F̄B = respond(model; F = false, B = true)

    # probability of F∨B
    p_ForB = respond_disjunction(model; F = true, B = true)

    z = compute_identities(p_F, p_B, p_FB, p_F̄B, p_FB̄, p_F̄B̄, p_ForB)
    return z
end

function compute_identities(p_F, p_B, p_FB, p_F̄B, p_FB̄, p_F̄B̄, p_ForB)
    z = zeros(6)
    z[1] = p_F + p_F̄B - p_ForB
    z[2] = p_B + p_FB̄ - p_ForB
    z[3] = p_FB + p_FB̄ - p_F
    z[4] = p_FB + p_F̄B - p_B
    z[5] = p_FB + p_F̄B + p_FB̄ - p_ForB
    z[6] = p_FB + p_F̄B + p_FB̄ + p_F̄B̄ - 1
    return z
end

function simulate_group(
    model_type::Type{<:AbstractACTRCPI},
    n_subj;
    func = predict_expected_identities,
    Θ,
    blc_dist,
    δ_dist,
    τ_dist
)
    output = zeros(6, n_subj)
    for s ∈ 1:n_subj
        # base level constant activation for each chunk 
        # a∧b, ā∧b, a∧̄b, ā∧b̄
        blcs = set(blc_dist, 4)
        blcs[end] = 0.0
        δ = set(δ_dist)
        τ = set(τ_dist)
        model = model_type(; blcs, Θ..., δ, τ)
        pred = func(model)
        pred[5] /= 2
        pred[6] /= 2
        output[:, s] = pred
    end
    return mean(output, dims = 2)[:]
end

"""
    simulate_group(
        model_type::Type{<:AbstractACTRCPI},
        n_subj,
        n_reps;
        func = predict_expected_identities,
        Θ,
        blc_dist,
        δ_dist,
        τ_dist
    )

# Arguments

- `model_type::Type{<:AbstractACTRCPI}`: model type (ACTRCPI or ACTRCPIAveraging)
- `n_subj`: number of simulated subjects in simulation 
- `n_reps`: number of times the simulation is repeated

# keywords

- `func = predict_expected_identities`: computes predicted values 
- `Θ`: fixed ACT-R parameters 
- `blc_dist`: a distribution or constant for base-level constants 
- `δ_dist`: a distribution or constant for partial matching 
- `τ_dist`: a distribution or constant for retrieval threshold 

# Returns 

- `z::Array`: identities averaged across simlated subjects. rows represent averaged identities and columns represent repetitions
"""
function simulate_group(
    model_type::Type{<:AbstractACTRCPI},
    n_subj,
    n_reps;
    func = predict_expected_identities,
    Θ,
    blc_dist,
    δ_dist,
    τ_dist
)
    z = fill(0.0, 6, n_reps)
    for r ∈ 1:n_reps
        z[:, r] = simulate_group(
            model_type,
            n_subj;
            func,
            Θ,
            blc_dist,
            δ_dist,
            τ_dist
        )
    end
    return z
end

set(x::Distribution) = rand(x)
set(x::Real) = x
set(x::Distribution, n::Int) = rand(x, n)
set(x::Real, n::Int) = fill(x, n)
