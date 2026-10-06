"""
    ACTRUnpacking{A}

# Fields 

- `actr::A`: an ACT-R model object 
- `n_sub_events`: the number of subevents per cause
- `n_retrievals`: the number of retrievals when simulating the judgment

# Constructors 

    ACTRUnpacking(; n_retrievals = 0, n_sub_events, bl, Θ...)

# Keywords

- `n_sub_events`: the number of subevents per cause
- `bl`: base level constant
- `Θ...`: optional keyword arguments for ACT-R model object 
"""
mutable struct ACTRUnpacking{A}
    actr::A
    n_sub_events1::Int
    n_retrievals::Int
end

function ACTRUnpacking(;
    n_retrievals = 0,
    n_sub_events1,
    n_sub_events2 = n_sub_events1,
    bl,
    Θ...
)
    chunks = populate_memory(ACTRUnpacking; n_sub_events1, n_sub_events2, bl)
    actr = ACTR(; declarative = Declarative(; memory = chunks), Θ...)
    return ACTRUnpacking(actr, n_sub_events1, n_retrievals)
end

"""
    populate_memory(
        ::Type{<:ACTRUnpacking};
        n_sub_events1,
        n_sub_events2 = n_sub_events1,
        bl
    )

Generates chunks for an ACT-R model of unpacking effects. 

# Arguments

- `::Type{<:ACTRUnpacking}`: an unpacking effect model type 

# Keywords
- `n_sub_events1`: the number of sub-events for the cancer category (e.g., sub-types of cancer death)
- `n_sub_events2 = n_sub_events1`: the number of sub-events for the non-cancer category (non-cancer types of death)
- `bl`: a distribution or constant for base level constants
"""
function populate_memory(
    ::Type{<:ACTRUnpacking};
    n_sub_events1,
    n_sub_events2 = n_sub_events1,
    bl
)
    n_sub_events = n_sub_events1 + n_sub_events2
    chunks = [
        Chunk(; cause = :cancer, type = i, bl = set(bl)) for
        i ∈ 1:n_sub_events1
    ]
    push!(
        chunks,
        [
            Chunk(; cause = :non_cancer, type = i, bl = set(bl)) for
            i ∈ (n_sub_events1 + 1):n_sub_events
        ]...
    )
    return chunks
end

"""
    judge(model::ACTRUnpacking; request...)

Judges the probability of events identified in `request...`.
`request...` serves as a filter for the numerator and as a retrieval request 

# Arguments 

- `model::ACTRUnpacking`: an ACT-R model of subadditivity
- `request...`: optional keyword arguments serving as a retrieval request
"""
function judge(model::ACTRUnpacking; request...)
    target = get_chunks(model.actr; request...)
    p, rf = retrieval_prob(model.actr, target; request...)
    # include retrieval failure
    return p + rf * 0.50
end

function respond(model::ACTRUnpacking; request...)
    (; n_retrievals) = model
    θ = judge(model; request...)
    return rand(Binomial(n_retrievals, θ)) / n_retrievals
end

"""
    simulate_trial(model::ACTRUnpacking)

Simulate a single trial in which the model outputs the unpacking factor based on two judgments:

1. judge the probability of cancer 
2. judge and sum the probability of each type of cancer

# Arguments

- `model::ACTRUnpacking`: an ACT-R model of subadditivity

# Keywords 

- `response_func = judge`: response function (judge or respond)

# Returns 

- `unpacking_factor`: the ratio of probability judgments in the unpacked over packed conditions.
"""
function simulate_trial(model::ACTRUnpacking; response_func = judge)
    n = model.n_sub_events1
    p_packed = judge(model; cause = :cancer)
    p_unpacked = 0.0
    for i ∈ 1:n
        p_unpacked += response_func(model; cause = :cancer, type = i)
    end
    return p_unpacked / p_packed
end
