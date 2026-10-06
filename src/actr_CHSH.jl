"""
    ACTRCHSH{A} <: DiscreteMultivariateDistribution

An ACT-R model of the CHSH inequality and marginal invariance

# Fields 

- `actr::A`: an ACT-R model object 
"""
mutable struct ACTRCHSH{A} <: DiscreteMultivariateDistribution
    actr::A
end

"""
    ACTRCHSH(; blcs, Θ...)

Creates an ACT-R model of the CHSH inequality and marginal invariance


# Fields 

- `blcs`: base-level constants for the 16 joint events 
- `Θ...`: optional `NamedTuple` of additional ACT-R parameters
"""
function ACTRCHSH(; blcs, Θ...)
    chunks = populate_memory(ACTRCHSH; blcs)
    actr = ACTR(; declarative = Declarative(; memory = chunks), Θ...)
    return ACTRCHSH(actr)
end

"""
    populate_memory(::Type{<:ACTRCHSH}; blcs)

Populates declarative memory with chunks for the ACT-R order effect model

# Arguments

- `::Type{<:ACTRCHSH}`: an ACT-R model for the CHSH inequality and marginal invariance

# Keywords

- `blcs`: base-level constants

# Output 

`chunks::Vector{<:Chunk}`: a vector of chunks with the following slot values:

 1.  (agreeable = 1, honest = 1, intelligent = 1, unusual = 1)
 2.  (agreeable = 0, honest = 1, intelligent = 1, unusual = 1)
 3.  (agreeable = 1, honest = 0, intelligent = 1, unusual = 1)
 4.  (agreeable = 0, honest = 0, intelligent = 1, unusual = 1)
 5.  (agreeable = 1, honest = 1, intelligent = 0, unusual = 1)
 6.  (agreeable = 0, honest = 1, intelligent = 0, unusual = 1)
 7.  (agreeable = 1, honest = 0, intelligent = 0, unusual = 1)
 8.  (agreeable = 0, honest = 0, intelligent = 0, unusual = 1)
 9.  (agreeable = 1, honest = 1, intelligent = 1, unusual = 0)
 10. (agreeable = 0, honest = 1, intelligent = 1, unusual = 0)
 11. (agreeable = 1, honest = 0, intelligent = 1, unusual = 0)
 12. (agreeable = 0, honest = 0, intelligent = 1, unusual = 0)
 13. (agreeable = 1, honest = 1, intelligent = 0, unusual = 0)
 14. (agreeable = 0, honest = 1, intelligent = 0, unusual = 0)
 15. (agreeable = 1, honest = 0, intelligent = 0, unusual = 0)
 16. (agreeable = 0, honest = 0, intelligent = 0, unusual = 0)
"""
function populate_memory(::Type{<:ACTRCHSH}; blcs)
    bool = [true, false]
    chunks = [
        Chunk(; agreeable = a, honest = h, intelligent = i, unusual = u) for
        a ∈ bool, h ∈ bool, i ∈ bool, u ∈ bool
    ][:]
    for i ∈ 1:length(blcs)
        chunks[i].bl = blcs[i]
    end
    return chunks
end

"""
    compute_joint_prob(model::ACTRCHSH; event, request...)

Computes the joint probability of an event based on a retrieval request.

# Arguments

- `model::ACTRCHSH`: an ACT-R model of the CHSH inequality and marginal invariance

# Keywords

- `event`: a `NamedTuple` for the target event for which the response probability is computed
- `request`: a `NamedTuple` for the retrieval request 
"""
function compute_joint_prob(model::ACTRCHSH; event, request...)
    (; actr) = model
    chunks = get_chunks(model.actr; event...)
    prob, _ = retrieval_prob(actr, chunks; request...)
    return prob
end

"""
    compute_joint_probs(model::ACTRCHSH; slot1, slot2)

Computes the joint response probability distribution for the specified question order. 

# Arguments

- `model::ACTRCHSH`: an ACT-R model of the CHSH inequality and marginal invariance

# Keywords

- `slot1`: the slot for the first attribute
- `slot2`: the slot for the second attribute

# Returns 

- `joint_probs`: a vector of joint probabilities where elements correspond to:
    1. (slot1 true, slot2 true)
    2. (slot1 false, slot2 true)
    3. (slot1 true, slot2 false)
    4. (slot1 false, slot2 false)
"""
function compute_joint_probs(model::ACTRCHSH; slot1, slot2)
    p = zeros(4)
    request = (slot1 => true, slot2 => true)
    p[1] = compute_joint_prob(model; event = (slot1 => true, slot2 => true), request...)
    p[2] = compute_joint_prob(model; event = (slot1 => false, slot2 => true), request...)
    p[3] = compute_joint_prob(model; event = (slot1 => true, slot2 => false), request...)
    p[4] = compute_joint_prob(model; event = (slot1 => false, slot2 => false), request...)
    return p
end

"""
    learn!(model::ACTRCHSH, n_events, joint_events, joint_prob_dist, time_dist)

The model learns the distribution of inhabitant attributes in a simulated environment.

# Arguments

- `model::ACTRCHSH`: an ACT-R model of the CHSH inequality and marginal invariance 
- `n_events`: the number of events from which the model learns about the environment 
- `joint_events`: a vector of slot-value pairs representing the properities of experienced events, i.e., the attributes of inhabitants 
- `joint_prob_dist`: a Dirichelet distribution defining joint probabilities over the four binary attributes 
- `time_dist`: an inter-arrival distribution between events 
"""
function learn!(model::ACTRCHSH, n_events, joint_events, joint_prob_dist, time_dist)
    θ = rand(joint_prob_dist)
    for _ ∈ 1:n_events
        idx = rand(Categorical(θ))
        joint_event = joint_events[idx]
        tΔ = rand(time_dist)
        increment_time!(model, tΔ)
        add_chunk!(model.actr; joint_event...)
    end
    return nothing
end

get_ev_prod(p) = (p[1] + p[4]) - (p[2] + p[3])

""" 
    get_chsh_test(p1, p2, p3, p4) 

Computes the test value for the CHSH inequality.

# Arguments

- `p1`: the joint distribution for context 1, with elements corresponding to (TT), (FT), (TF), (FF)
- `p2`: the joint distribution for context 2, with elements corresponding to (TT), (FT), (TF), (FF)
- `p3`: the joint distribution for context 3, with elements corresponding to (TT), (FT), (TF), (FF)
- `p4`: the joint distribution for context 4, with elements corresponding to (TT), (FT), (TF), (FF)
"""
function get_chsh_test(p1, p2, p3, p4)
    x1 = get_ev_prod(p1)
    x2 = get_ev_prod(p2)
    x3 = get_ev_prod(p3)
    x4 = get_ev_prod(p4)
    return maximum([
        abs(x1 + x2 + x3 - x4),
        abs(x1 + x2 - x3 + x4),
        abs(x1 - x2 + x3 + x4),
        abs(-x1 + x2 + x3 + x4)
    ])
end

"""
    get_chsh_test(probs::Dict)

Computes test values for all four CHSH inequalities 

# Arguments 

- `probs::Dict`: a dictionary of 6 joint probabilities based on 4 random variables 

# Output 

- `chsh`: the absolute maximum of the following variations of chsh

1. A * I + H * I + H * U - A * U
2. A * I + H * I - H * U + A * U
3. A * I - H * I + H * U + A * U
4. -A * I + H * I + H * U + A * U
"""
function get_chsh_test(probs::Dict)
    # A * I + H * I + H * U - A * U
    return get_chsh_test(
        probs[[:agreeable, :intelligent]],
        probs[[:honest, :intelligent]],
        probs[[:honest, :unusual]],
        probs[[:agreeable, :unusual]]
    )
end

"""
    compute_marginals(joint_probs::Dict, event)

Computes all marginal probabilities for a set of joint probabilities. 

- `probs::Dict`: a dictionary of 6 joint probabilities based on 4 random variables. Each joint probability vector has elements:
    1. (slot1 true, slot2 true)
    2. (slot1 false, slot2 true)
    3. (slot1 true, slot2 false)
    4. (slot1 false, slot2 false)
- `event`: the event for the marginal probability 
"""
function compute_marginals(joint_probs::Dict, event)
    marginals = Float64[]
    for (k, v) ∈ joint_probs
        if k[1] == event
            marginal = v[1] + v[3]
            push!(marginals, marginal)
        elseif k[2] == event
            marginal = v[1] + v[2]
            push!(marginals, marginal)
        end
    end
    return marginals
end

"""
    make_joint_event_probs(p)

Creates a vector of 16 joint probabilities over four binary random variables. The marginal probabilities of the four events 
are independent. 

# Arguments

- `p`: the marginal probability for each of the four events 
"""
function make_joint_event_probs(p)
    ps = [p, (1-p)]
    return prod.(Base.product(fill(ps, 4)...))[:]
end

"""
    make_dirichlet(p, n)

Creates a Dirchlet distribution object for the joint probability over four binary events. Two simplifying assumptions are made:

    1. The marginal probabilities for each event are equal 
    2. The events are independent (on average)  

# Arguments

- `p`: the marginal probability for each of the four events 
- `n`: the concentration parameter, where larger values decrease variance in the joint probability distributions
"""
function make_dirichlet(p, n)
    ps = make_joint_event_probs(p)
    return Dirichlet(ps * n)
end
