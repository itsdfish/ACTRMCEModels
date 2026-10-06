"""
    actr_order_effect(y, n, config, Θ)

An ACT-R model of order effects for Bayesian parameter estimation.

# Arguments 

- `y`: frequency data for each order. The first vector corresponds to the Clinton-Gore joint response distribution. The second vector corresponds to the 
    Gore-Clinton joint response distribution. Elements in each sub-vector correspond to frequencies for
    (Clinton honest, Gore honest), (Clinton not honest, Gore honest), (Clinton honest, Gore not honest), (Clinton not honest, Gore not honest)
- `chunk_counts`: a `NamedTuple` for chunk counts 
- `Θ`: a `NamedTuple` of additional ACT-R parameters 
"""
@model function actr_order_effect(y, n, chunk_counts, Θ)
    # data = (y,n)
    bl_ch ~ Normal(0, 0.5)
    bl_ch̄ ~ Normal(0, 0.5)
    bl_gh ~ Normal(0, 0.5)
    bl_gh̄ = zero(bl_ch)
    γ ~ Uniform(2, 5)
    model = ACTROrderEffect(; chunk_counts..., bl_ch, bl_ch̄, bl_gh, bl_gh̄, Θ..., γ)
    p_clinton_gore = compute_joint_probs(model; clinton_gore = true)
    y[1] ~ Multinomial(n[1], p_clinton_gore)
    p_gore_clinton = compute_joint_probs(model; clinton_gore = false)
    y[2] ~ Multinomial(n[2], p_gore_clinton)
    # data ~ ACTROrderEffect(; chunk_counts..., bl_ch, bl_ch̄, bl_gh,bl_gh̄, Θ..., γ)
    return (; bl_ch, bl_ch̄, bl_gh, bl_gh̄, γ)
end
