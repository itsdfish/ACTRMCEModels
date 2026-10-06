"""
    ACTROrderEffect{A, T} <: DiscreteMultivariateDistribution

An ACT-R model of order effects. 

# Fields 

- `actr::A`: an ACT-R model object 
- `t::T`: an otherwise unused variable for automatic differentiation
"""
mutable struct ACTROrderEffect{A, T} <: DiscreteMultivariateDistribution
    actr::A
    t::T
end

"""
    ACTROrderEffect(; n_ch, n_ch̄, n_gh, n_gh̄, bl_ch, bl_ch̄, bl_gh, bl_gh̄, γ, Θ...)

An ACT-R model of order effects. 

# Fields 

- `n_ch`: number of chunks for Clinton honest
- `n_ch̄`: number of chunks for Clinton not honest
- `n_gh`: number of chunks for Gore honest
- `n_gh̄`: number of chunks for Gore not honest
- `bl_ch`: base-level constant for Clinton honest
- `bl_ch̄`: base-level constant for Clinton not honest
- `bl_gh`: base-level constant for Gore honest
- `bl_gh̄`: base-level constant for Gore not honest
- `γ`: maximum association parameter 
- `Θ...`: optional `NamedTuple` of additional ACT-R parameters
"""
function ACTROrderEffect(; n_ch, n_ch̄, n_gh, n_gh̄, bl_ch, bl_ch̄, bl_gh, bl_gh̄, γ, Θ...)
    chunks = populate_memory(
        ACTROrderEffect;
        n_ch,
        n_ch̄,
        n_gh,
        n_gh̄,
        bl_ch,
        bl_ch̄,
        bl_gh,
        bl_gh̄
    )

    _γ = set(γ)
    actr = ACTR(;
        declarative = Declarative(; memory = chunks),
        imaginal = Imaginal(; buffer = Chunk[]),
        γ = _γ,
        Θ...
    )
    return ACTROrderEffect(actr, _γ)
end

rand(model::ACTROrderEffect, n1::Int, n2::Int) = rand(Random.default_rng(), model, n1, n2)

"""
    rand(model::ACTROrderEffect, n1::Int, n2::Int)

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 
- `n1`: the number of observations for the Clinton-Gore joint response distribution 
- `n2`: the number of observations for the Gore-Clinton joint response distribution 
"""
function rand(rng::AbstractRNG, model::ACTROrderEffect, n1::Int, n2::Int)
    p_clinton_gore = compute_joint_probs(model; clinton_gore = true)
    data1 = rand(rng, Multinomial(n1, p_clinton_gore))
    p_gore_clinton = compute_joint_probs(model; clinton_gore = false)
    data2 = rand(rng, Multinomial(n2, p_gore_clinton))
    return [data1, data2]
end

"""
    logpdf(model::ACTROrderEffect, data::Tuple)

Computes the loglikelihood of the data for the ACT-R order effect model

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 
- `data::Tuple`: element one contains joint responses for Clinton-Gore order and element 2 contains joint responses for the 
    Gore-Clinton order 
"""
function logpdf(model::ACTROrderEffect, data::Tuple)
    p_clinton_gore = compute_joint_probs(model; clinton_gore = true)
    LL = logpdf(Multinomial(data[2][1], p_clinton_gore), data[1][1])
    p_gore_clinton = compute_joint_probs(model; clinton_gore = false)
    LL += logpdf(Multinomial(data[2][2], p_gore_clinton), data[1][2])
    return LL
end

loglikelihood(dist::ACTROrderEffect, data::Tuple) = logpdf(dist, data)

Base.broadcastable(x::ACTROrderEffect) = Ref(x)

length(d::ACTROrderEffect) = 1

"""
    populate_memory(::Type{<:ACTROrderEffect};
        n_ch,
        n_ch̄,
        n_gh,
        n_gh̄,
        bl_ch = 0.0,
        bl_ch̄ = 0.0,
        bl_gh = 0.0,
        bl_gh̄ = 0.0
    )

Populates declarative memory with chunks for the ACT-R order effect model

# Arguments

- `::Type{<:ACTROrderEffect}`: a type for the ACT-R order effect model 

# Keywords

- `n_ch`: number of chunks indicating Clinton is honest 
- `n_ch̄`: number of chunks indicating Clinton is not honest
- `n_gh`: number of chunks indicating Gore is honest
- `n_gh̄`: number of chunks indicating Gore is not honest
- `bl_ch = 0.0`: the base-level constant for chunks in which Clinton was honest 
- `bl_ch̄ = 0.0`: the base-level constant for chunks in which Clinton not was honest 
- `bl_gh = 0.0`: the base-level constant for chunks in which Gore was honest 
- `bl_gh̄ = 0.0`: the base-level constant for chunks in which Gore not was honest 
"""
function populate_memory(::Type{<:ACTROrderEffect};
    n_ch,
    n_ch̄,
    n_gh,
    n_gh̄,
    bl_ch = 0.0,
    bl_ch̄ = 0.0,
    bl_gh = 0.0,
    bl_gh̄ = 0.0
)
    chunks = [
        Chunk(; name = :Clinton, honest = true, statement = rand(), bl = set(bl_ch))
        for
        i ∈ 1:set(n_ch)
    ]
    push!(
        chunks,
        [
            Chunk(; name = :Clinton, honest = false, statement = rand(), bl = set(bl_ch̄))
            for
            i ∈ 1:set(n_ch̄)
        ]...
    )
    push!(
        chunks,
        [
            Chunk(; name = :Gore, honest = true, statement = rand(), bl = set(bl_gh)) for
            i ∈ 1:set(n_gh)
        ]...
    )
    push!(
        chunks,
        [
            Chunk(; name = :Gore, honest = false, statement = rand(), bl = set(bl_gh̄)) for
            i ∈ 1:set(n_gh̄)
        ]...
    )
    return chunks
end

"""
    compute_joint_prob(model::ACTROrderEffect; request1, request2)

Computes the joint response probability based on request1 and request2. 

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 

# Keywords

- `request1`: a `NamedTuple` for the retrieval request to the first question 
- `request2`: a `NamedTuple` for the retrieval request to the second question 
"""
function compute_joint_prob(model::ACTROrderEffect; request1, request2)
    (; actr) = model
    empty!(actr.imaginal.buffer)
    target = retrieval_request(actr; request1...)
    p_a, _ = retrieval_prob(actr, target; name = request1.name)

    imaginal_chunk = get_chunks(actr; request1...)
    isempty(imaginal_chunk) ? nothing : push!(actr.imaginal.buffer, imaginal_chunk[1])

    target = retrieval_request(actr; request2...)
    p_b, _ = retrieval_prob(actr, target; name = request2.name)

    return p_a * p_b
end

"""
    compute_joint_probs(model::ACTROrderEffect{A, T}; clinton_gore) where {A, T}

Computes the joint response probability distribution for the specified question order. 

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 

# Keywords

- `clinton_gore`: computes joint probability distribution for the Clinton-Gore order if true. Computes the joint probability distribution for the Gore-Clinton order 
    otherwise

# Returns 

- `joint_probs`: a vector of joint probabilities where elements correspond to  (Clinton honest, Gore honest), (Clinton not honest, Gore honest), (Clinton honest, Gore not honest), (Clinton not honest, Gore not honest)
"""
function compute_joint_probs(model::ACTROrderEffect{A, T}; clinton_gore) where {A, T}
    p = zeros(T, 4)
    p[1] = compute_joint_prob(
        model;
        request1 = clinton_gore ? (; name = :Clinton, honest = true) :
                   (; name = :Gore, honest = true),
        request2 = clinton_gore ? (; name = :Gore, honest = true) :
                   request2 = (; name = :Clinton, honest = true)
    )

    p[2] = compute_joint_prob(
        model;
        request1 = clinton_gore ? (; name = :Clinton, honest = false) :
                   (; name = :Gore, honest = true),
        request2 = clinton_gore ? (; name = :Gore, honest = true) :
                   (; name = :Clinton, honest = false)
    )

    p[3] = compute_joint_prob(
        model;
        request1 = clinton_gore ? (; name = :Clinton, honest = true) :
                   (; name = :Gore, honest = false),
        request2 = clinton_gore ? (; name = :Gore, honest = false) :
                   (; name = :Clinton, honest = true)
    )

    p[4] = compute_joint_prob(
        model;
        request1 = clinton_gore ? (; name = :Clinton, honest = false) :
                   (; name = :Gore, honest = false),
        request2 = clinton_gore ? (; name = :Gore, honest = false) :
                   (; name = :Clinton, honest = false)
    )
    return p
end

"""
    compute_q1(model::ACTROrderEffect)

Computes the q-value. 

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 
"""
function compute_q1(model::ACTROrderEffect)
    pchgh = compute_joint_prob(
        model;
        request1 = (; name = :Clinton, honest = true),
        request2 = (; name = :Gore, honest = true)
    )

    pghch = compute_joint_prob(
        model;
        request1 = (; name = :Gore, honest = true),
        request2 = (; name = :Clinton, honest = true)
    )

    pch̄gh̄ = compute_joint_prob(
        model;
        request1 = (; name = :Clinton, honest = false),
        request2 = (; name = :Gore, honest = false)
    )

    pgh̄ch̄ = compute_joint_prob(
        model;
        request1 = (; name = :Gore, honest = false),
        request2 = (; name = :Clinton, honest = false)
    )
    return (pchgh - pghch) + (pch̄gh̄ - pgh̄ch̄)
end

function compute_q2(model::ACTROrderEffect)
    pchgh̄ = compute_joint_prob(
        model;
        request1 = (; name = :Clinton, honest = true),
        request2 = (; name = :Gore, honest = false)
    )

    pgh̄ch = compute_joint_prob(
        model;
        request1 = (; name = :Gore, honest = false),
        request2 = (; name = :Clinton, honest = true)
    )

    pch̄gh = compute_joint_prob(
        model;
        request1 = (; name = :Clinton, honest = false),
        request2 = (; name = :Gore, honest = true)
    )

    pghch̄ = compute_joint_prob(
        model;
        request1 = (; name = :Gore, honest = true),
        request2 = (; name = :Clinton, honest = false)
    )
    return (pchgh̄ - pgh̄ch) + (pch̄gh - pghch̄)
end

"""
    compute_order_effects(model::ACTROrderEffect)

Computes the joint response probability distribution for the specified question order. 

# Arguments

- `model::ACTROrderEffect`: an ACT-R model of order effects 

# Returns 

- `order_effects`: a vector of order effects probabilities where elements correspond to (Clinton honest, Gore honest), (Clinton not honest, Gore honest), (Clinton honest, Gore not honest), (Clinton not honest, Gore not honest)
    Order effects are computed as Clinton-Gore minus Gore-Clinton 
"""
function compute_order_effects(model::ACTROrderEffect)
    p_cg = compute_joint_probs(model; clinton_gore = true)
    p_gc = compute_joint_probs(model; clinton_gore = false)
    return p_cg - p_gc
end

function compute_order_effects(x::Vector)
    n = sum.(x)
    p = x ./ n
    return p[1] .- p[2]
end

function compute_q1(x::Vector)
    order_effect = compute_order_effects(x)
    return order_effect[1] + order_effect[4]
end
