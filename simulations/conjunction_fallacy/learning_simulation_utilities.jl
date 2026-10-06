"""
    run_learning_sim(; blcs, n_events, event_set, time_dist, event_dist, Θ...)

The model repeatedly learns the environment and then makes judgments for events B, F, and F∧B. 

# Arguments

- `response_func = judge`: response function (judge or respond)
- `blcs`: base-level constants for each chunk
- `n_reps`: number of times the learning and judgment simulation is performed
- `n_events`: the number of events from which the model learns about the environment 
- `event_set`: a vector of slot-value pairs representing the properities of experienced events 
- `time_dist`: an inter-event time distribution 
- `event_dist`: a distribution over `event_set` 
- `Θ...`: an optional `NamedTuple` of ACT-R parameters 
"""

function run_learning_sim(;
    response_func = judge,
    blcs,
    n_reps,
    n_events,
    event_set,
    time_dist,
    event_dist,
    Θ...
)
    results = map(
        _ -> run_learning_trial(;
            response_func,
            blcs,
            Θ,
            n_events,
            event_set,
            time_dist,
            event_dist
        ),
        1:n_reps
    )
    return stack(results, dims = 1)
end

"""
    run_learning_trial(; blcs, Θ, n_events, event_set, time_dist, event_dist)

The model learns the environment and then makes judgments for events B, F, and F∧B. 

# Arguments

- `response_func = judge`: response function (judge or respond)
- `blcs`: base-level constants for each chunk
- `Θ`: a `NamedTuple` of ACT-R parameters 
- `n_events`: the number of events from which the model learns about the environment 
- `event_set`: a vector of slot-value pairs representing the properities of experienced events 
- `time_dist`: an inter-event time distribution 
- `event_dist`: a distribution over `event_set` 
"""
function run_learning_trial(;
    response_func = judge,
    blcs,
    Θ,
    n_events,
    event_set,
    time_dist,
    event_dist
)
    model = ACTRCPI(; blcs, Θ...)
    learn!(model, n_events, event_set, time_dist, event_dist)
    increment_time!(model, 50)

    # probability of B
    p_B = response_func(model; B = true)

    # probability of F
    p_F = response_func(model; F = true)

    # probability of F ∧ B
    p_FB = response_func(model; F = true, B = true)

    return [p_B, p_FB, p_F]
end
