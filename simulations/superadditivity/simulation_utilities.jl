"""
    simulate_individual(; response_func, n_sim, n_sub_events1, bl, τ, δ, Θ...)

Simulates an individual multiple times and returns a vector of unpacking factors 

# Keywords

- `response_func`: response function (judge or respond)
- `n_sim`: number of simulations (replications) performed 
- `n_sub_events1`: the number of sub-events for the cancer category
- `bl`: distribution or constant for base level constants 
- `τ`: retrieval threshold
- `δ`: mismatch penalty
- `Θ...`: NamedTuple of optional default ACT-R parameters
"""
function simulate_individual(; response_func, n_sim, n_sub_events1, bl, τ, δ, Θ...)
    unpacking_factors = fill(0.0, n_sim)
    model = ACTRUnpacking(; n_sub_events1, bl, Θ..., τ, δ)
    for i ∈ 1:n_sim
        unpacking_factors[i] = simulate_trial(model; response_func)
    end
    return unpacking_factors
end

"""
    simulate_group(
        n_sim,
        n_subj;
        τ_dist,
        δ_dist,
        blc_dist,
        sub_event_dist,
        response_func,
        Θ
    )

Simulates sub-or-super-additivity in a group of individuals. 

# Arguments 

- `n_sim`: number of simulations (replications) performed 
- `n_subj`: number of subjects in each simulation 

# Keywords

- `τ_dist`: distribution or constant for τ (retrieval threshold)
- `δ_dist`: distribution or constant for δ (mismatch penalty)
- `blc_dist`: distribution or constant for base level constants 
- `sub_event_dist`: distribution or constant for number of sub-events in each category.
- `response_func`: response function (judge or respond)
- `Θ`: NamedTuple of default ACT-R parameters

# Returns 

- `unpacking_factor`: a vector of vectors where the inner vectors correspond to unpacking factors for each individual, e.g., uf[1] is a vector 
of unpacking factors for the first individual.
"""
function simulate_group(
    n_sim,
    n_subj;
    τ_dist,
    δ_dist,
    blc_dist,
    sub_event_dist,
    response_func,
    Θ
)
    results = map(1:n_subj) do _
        τ = set(τ_dist)
        n_sub_events1 = set(sub_event_dist)
        δ = set(δ_dist)
        # base level constant for each chunk is a distribution 
        bl = blc_dist
        return simulate_individual(; response_func, n_sim, n_sub_events1, bl, Θ..., τ, δ)
    end
    return results
end

"""
    mean_unpacking_factors(
        n_subj;
        τ_dist,
        δ_dist,
        blc_dist,
        sub_event_dist,
        response_func,
        Θ
    )

Generates a group of individuals from specified parameter distributions and returns the mean unpacking factor across individuals. 

# Arguments 

- `n_subj`: number of subjects in each simulation 

# Keywords

- `τ_dist`: distribution or constant for τ (retrieval threshold)
- `δ_dist`: distribution or constant for δ (mismatch penalty)
- `blc_dist`: distribution or constant for base level constants 
- `sub_event_dist`: distribution or constant for number of sub-events in each category.
- `response_func`: response function (judge or respond)
- `Θ`: NamedTuple of default ACT-R parameters

# Returns 

- `mean_unpacking_factor`: the mean unpacking factor across individuals.
"""
function mean_unpacking_factors(
    n_subj;
    τ_dist,
    δ_dist,
    blc_dist,
    sub_event_dist,
    response_func,
    Θ
)
    ufs = simulate_group(
        1,
        n_subj;
        τ_dist,
        δ_dist,
        blc_dist,
        sub_event_dist,
        response_func,
        Θ
    )
    return mean(ufs)[1]
end
