"""
SEIHCVRD age-specific transition probabilities and durations.
The age structure is read from disk.

Pr(next state is j | current state is i, age) = 1 / (1 + exp(-(a0 + a1*age)))

duration|age ~ Poisson(lambda(age)), where lambda(age) = exp(b0 + b1*age)
"""
module abm

export init_model

using DataFrames
using Distributions

using ..core
using ..contacts

function init_model(indata::Dict{String, DataFrame}, params::T, cfg) where {T <: NamedTuple}
    # Init model
    agedist   = indata["age_distribution"]
    npeople   = sum(agedist[!, :Count])
    agents    = Vector{Person}(undef, npeople)
    schedule0 = Tuple{Function, Int}[]
    schedule  = [Tuple{Function, Int}[] for t = 1:(cfg.maxtime - 1)]
    model     = Model(agents, params, 0, cfg.maxtime, schedule0, schedule)

    # Construct people
    id = 0
    cdf0 = cumsum(cfg.initial_state_distribution)
    n_agegroups = size(agedist, 1)
    for i = 1:n_agegroups
        agegroup = String(agedist[i, :AgeGroup])  # Example: "Age 0-4", "Age 85+"
        agegroup = agegroup[5:end]                # Example: "0-4", "85+"
        idx = findfirst(==('-'), agegroup)
        idx = isnothing(idx) ? findfirst(==('+'), agegroup) : idx
        lb  = parse(Int, agegroup[1:(idx-1)])
        ub  = lb + 4  # Oldest age is therefore 89
        n_agegroup = agedist[i, :Count]
        for j in 1:n_agegroup
            id += 1
            age = rand(lb:ub)
            agents[id] = Person(id, model, cdf0, age)
        end
    end

    # Sort agents and populate contacts
    populate_contacts!(agents, cfg, indata)
    model
end

mutable struct Person <: AbstractAgent
    id::Int
    status::Symbol  # S, E, I, H, C, V, R, D

    # Contacts
    household::Vector{Int}  # People in the same household
    work::Vector{Int}       # Child care, school, university, workplace
    community::Vector{Int}  # Shops, transport, pool, library, etc
    social::Vector{Int}     # Family and/or friends outside the household

    # Risk factors
    age::Int
end

function Person(id::Int, model, cdf0::Vector{Float64}, age::Int)
    params = model.params
    r = rand()
    if r <= cdf0[1]
        status = :S
    elseif r <= cdf0[2]
        status = :E
        dur    = dur_E(params, age)
        schedule!(id, dur, exit_E!, model)
    elseif r <= cdf0[3]
        status = :I
        dur    = dur_I(params, age)
        schedule!(id, dur, exit_I!, model)
    elseif r <= cdf0[4]
        status = :H
        dur    = dur_H(params, age)
        schedule!(id, dur, exit_H!, model)
    elseif r <= cdf0[5]
        status = :C
        dur    = dur_C(params, age)
        schedule!(id, dur, exit_C!, model)
    elseif r <= cdf0[6]
        status = :V
        dur    = dur_V(params, age)
        schedule!(id, dur, exit_V!, model)
    elseif r <= cdf0[7]
        status = :R
    else
        status = :D
    end
    Person(id, status, Int[], Int[], Int[], Int[], age)
end

function exit_E!(agent::Person, model, t, scenario)
    agent.status = :I
    infect_contacts!(agent, model, t, scenario)
    dur = dur_I(model.params, agent.age)
    schedule!(agent.id, t + dur, exit_I!, model)
end

function exit_I!(agent::Person, model, t, scenario)
    params = model.params
    age    = agent.age
    if rand() <= p_H(params, age)
        agent.status = :H
        dur = dur_H(params, age)
        schedule!(agent.id, t + dur, exit_H!, model)
    else
        agent.status = :R
    end
end

function exit_H!(agent::Person, model, t, scenario)
    params = model.params
    age    = agent.age
    if rand() <= p_C(params, age)
        agent.status = :C
        dur = dur_C(params, age)
        schedule!(agent.id, t + dur, exit_C!, model)
    else
        agent.status = :R
    end
end

function exit_C!(agent::Person, model, t, scenario)
    params = model.params
    age    = agent.age
    if rand() <= p_V(params, age)
        agent.status = :V
        dur = dur_V(params, age)
        schedule!(agent.id, t + dur, exit_V!, model)
    else
        agent.status = :R
    end
end

function exit_V!(agent::Person, model, t, scenario)
    agent.status = rand() <= p_D(model.params, agent.age) ? :D : :R
end

################################################################################
# Functions called by event functions

dur_E(params, age) = rand(Poisson(exp(params.b0_E + params.b1_E * age)))
dur_H(params, age) = rand(Poisson(exp(params.b0_H + params.b1_H * age)))
dur_I(params, age) = rand(Poisson(exp(params.b0_I + params.b1_I * age)))
dur_C(params, age) = rand(Poisson(exp(params.b0_C + params.b1_C * age)))
dur_V(params, age) = rand(Poisson(exp(params.b0_V + params.b1_V * age)))

"Pr(Next state is H | Current state is I, age)."
function p_H(params, age)
    age <= 9  && return params.p_H_0to9
    age <= 19 && return params.p_H_10to19
    age <= 29 && return params.p_H_20to29
    age <= 39 && return params.p_H_30to39
    age <= 49 && return params.p_H_40to49
    age <= 59 && return params.p_H_50to59
    age <= 69 && return params.p_H_60to69
    age <= 79 && return params.p_H_70to79
    params.p_H_gte80
end

p_C(params, age) = 1.0 / (1.0 + exp(-(params.a0_C + params.a1_C * age)))  # Pr(Next state is C | Current state is H, age)
p_V(params, age) = 1.0 / (1.0 + exp(-(params.a0_V + params.a1_V * age)))  # Pr(Next state is V | Current state is C, age)
p_D(params, age) = 1.0 / (1.0 + exp(-(params.a0_D + params.a1_D * age)))  # Pr(Next state is D | Current state is V, age)

p_infect(params, age) = 1.0 / (1.0 + exp(-(params.a0_infect + params.a1_infect * age)))  # Pr(Infect contact | age)

function infect_contacts!(agent::Person, model, t::Int, scenario)
    agents      = model.agents
    params      = model.params
    age         = agent.age
    pr_infect   = p_infect(params, age)
    p_workplace = age <= 17 ? scenario.school : scenario.workplace
    infect_household_contacts!(scenario.household, pr_infect, params, agent, agents, model, t)
    infect_workplace_contacts!(p_workplace,        pr_infect, params, agent, agents, model, t)
    infect_community_contacts!(scenario.community, pr_infect, params, agent, agents, model, t)
    infect_social_contacts!(scenario.social,       pr_infect, params, agent, agents, model, t)
end

function infect_household_contacts!(p_household, pr_infect, params, agent, agents, model, t)
    p_household == 0.0 && return
    for id in agent.household
        rand() > p_household && continue
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
end

function infect_workplace_contacts!(p_workplace, pr_infect, params, agent, agents, model, t)
    p_workplace == 0.0 && return
    for id in agent.work
        rand() > p_workplace && continue
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
end

function infect_community_contacts!(p_community, pr_infect, params, agent, agents, model, t)
    p_community == 0.0 && return
    for id in agent.community
        rand() > p_community && continue
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
end

function infect_social_contacts!(p_social, pr_infect, params, agent, agents, model, t)
    p_social == 0.0 && return
    for id in agent.social
        rand() > p_social && continue
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
end

"Infect contact with probability p_infect."
function infect_contact!(pr_infect::Float64, params::T, contact::Person, model, t::Int) where {T <: NamedTuple}
    if contact.status == :S
        if rand() <= pr_infect
            contact.status = :E
            dur = dur_E(params, contact.age)
            schedule!(contact.id, t + dur, exit_E!, model)
        end
    end
    #=
    elseif contact.status == :R
        if rand() <= contact.p_reinfection
            contact.status = :E
            dur = dur_E(params, contact.age)
            schedule!(contact.id, t + dur, :exit_E, model)
        end
    end
    =#
end

end