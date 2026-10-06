"""
    run_learning_sim(;
        blcs,
        n_reps,
        n_events,
        joint_events,
        time_dist,
        joint_prob_dist,
        Θ...
    )

The model repeatedly learns the distribution of inhabitant attributes in a simulated environment.

# Keywords

- `blcs`: base-level constants for the 16 joint events 
- `n_reps`: the number of repetitions of the simulation
- `n_events`: the number of events from which the model learns about the environment 
- `joint_events`: a vector of slot-value pairs representing the properities of experienced events, i.e., the attributes of inhabitants 
- `joint_prob_dist`: a Dirichelet distribution defining joint probabilities over the four binary attributes 
- `time_dist`: an inter-arrival distribution between events 
- `Θ...`: optional `NamedTuple` of additional ACT-R parameters
"""
function run_learning_sim(;
    blcs,
    n_reps,
    n_events,
    joint_events,
    time_dist,
    joint_prob_dist,
    Θ...
)
    results = map(
        _ -> run_learning_trial(;
            blcs,
            Θ,
            n_events,
            joint_events,
            joint_prob_dist,
            time_dist
        ),
        1:n_reps
    )
    stacked_results = (;
        sd_agreeable = map(x -> x.sd_agreeable, results),
        sd_honest = map(x -> x.sd_honest, results),
        sd_intelligent = map(x -> x.sd_intelligent, results),
        sd_unusual = map(x -> x.sd_unusual, results),
        chsh = map(x -> x.chsh, results),
        joint_probs = map(x -> x.joint_probs, results)
    )
    return stacked_results
end

"""
    run_learning_trial(; blcs, Θ, n_events, joint_events, joint_prob_dist, time_dist)
        
The model learns the distribution of inhabitant attributes in a simulated environment.

# Keywords

- `blcs`: base-level constants for the 16 joint events 
- `Θ`: optional `NamedTuple` of additional ACT-R parameters
- `n_events`: the number of events from which the model learns about the environment 
- `joint_events`: a vector of slot-value pairs representing the properities of experienced events, i.e., the attributes of inhabitants 
- `joint_prob_dist`: a Dirichelet distribution defining joint probabilities over the four binary attributes 
- `time_dist`: an inter-arrival distribution between events 
"""
function run_learning_trial(; blcs, Θ, n_events, joint_events, joint_prob_dist, time_dist)
    # initialize model 
    model = ACTRCHSH(; blcs, Θ...)
    # learn the events 
    learn!(model, n_events, joint_events, joint_prob_dist, time_dist)
    # increment time before making judgments 
    increment_time!(model, 50)
    # 6 combinations of joint event pairs out of 4
    contexts = combinations([:agreeable, :honest, :intelligent, :unusual], 2)
    # compute 2×2 joint response distribution over all 6 combinations 
    joint_probs =
        Dict(c => compute_joint_probs(model; slot1 = c[1], slot2 = c[2]) for c ∈ contexts)
    # compute the four marginal probabilities 
    sd_agreeable = compute_marginals(joint_probs, :agreeable) |> std
    sd_honest = compute_marginals(joint_probs, :honest) |> std
    sd_intelligent = compute_marginals(joint_probs, :intelligent) |> std
    sd_unusual = compute_marginals(joint_probs, :unusual) |> std
    # compute the chsh value (max absolute of 4 variations)
    chsh = get_chsh_test(joint_probs)
    return (; sd_agreeable, sd_honest, sd_intelligent, sd_unusual, chsh, joint_probs)
end
