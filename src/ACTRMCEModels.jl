module ACTRMCEModels

using ACTRModels
using Distributions
using LinearAlgebra
using Random

export ACTRCHSH
export ACTRCPI
export ACTRCPIAveraging
export ACTROrderEffect
export ACTRUnpacking
export QuantumJointProb

export compute_joint_prob
export compute_joint_probs
export compute_marginals
export compute_order_effects
export compute_prob_F
export compute_prob_FB
export compute_prob_B
export compute_q1
export compute_q2
export get_chsh_inequalities
export increment_time!
export judge
export judge_disjunction
export learn!
export make_dirichlet
export predict_expected_identities
export predict_noisy_identities
export respond
export simulate_group

import Distributions: length
import Distributions: logpdf
import Distributions: loglikelihood
import Distributions: rand

include("actr_joint_prob_model.jl")
include("quantum_joint_prob_model.jl")
include("actr_order_effect_model.jl")
include("actr_subadditivity_model.jl")
include("actr_CHSH.jl")
# Write your package code here.
end
