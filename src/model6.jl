"""
SEIHCVRD age-specific transition probabilities and durations.
The age structure is read from disk.

Pr(next state is j | current state is i, age) = 1 / (1 + exp(-(a0 + a1*age)))

duration|age ~ Poisson(lambda(age)), where lambda(age) = exp(b0 + b1*age)
"""
module model6

export init_model

using DataFrames
using Distributions

using ..core

function init_model(indata::Dict{String, DataFrame}, params::T, maxtime, dist0) where {T <: NamedTuple}
    agedist   = indata["age_distribution"]
    npeople   = sum(agedist[!, :Count])
    agents    = Vector{Person}(undef, npeople)
    schedule0 = EventBag()
    schedule  = [EventBag() for t = 1:(maxtime - 1)]
    model     = Model(agents, params, 0, maxtime, schedule0, schedule)
    cdf0      = cumsum(dist0)
    for id in 1:npeople
        agents[id] = Person(id, model, cdf0)
    end
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

function Person(id::Int, model, cdf0::Vector{Float64})
    age = 20
    params = model.params
    r = rand()
    if r <= cdf0[1]
        status = :S
    elseif r <= cdf0[2]
        status = :E
        dur    = dur_E(params, age)
        schedule!(id, dur, :exit_E, model)
    elseif r <= cdf0[3]
        status = :I
        dur    = dur_I(params, age)
        schedule!(id, dur, :exit_I, model)
    elseif r <= cdf0[4]
        status = :H
        dur    = dur_H(params, age)
        schedule!(id, dur, :exit_H, model)
    elseif r <= cdf0[5]
        status = :C
        dur    = dur_C(params, age)
        schedule!(id, dur, :exit_C, model)
    elseif r <= cdf0[6]
        status = :V
        dur    = dur_V(params, age)
        schedule!(id, dur, :exit_V, model)
    elseif r <= cdf0[7]
        status = :R
    else
        status = :D
    end
    npeople   = length(model.agents)
    household = Int[]
    work      = Int[]
    community = Int[]
    social    = sample(1:npeople, 35; replace=false)
    Person(id, status, household, work, community, social, age)
end

struct EventBag <: AbstractEventBag
    exit_E::Tuple{Function, Vector{Int}}
    exit_I::Tuple{Function, Vector{Int}}
    exit_H::Tuple{Function, Vector{Int}}
    exit_C::Tuple{Function, Vector{Int}}
    exit_V::Tuple{Function, Vector{Int}}
end

EventBag() = EventBag((exit_E!, Int[]), (exit_I!, Int[]), (exit_H!, Int[]), (exit_C!, Int[]), (exit_V!, Int[]))

function exit_E!(agent::Person, model, t)
    agent.status = :I
    infect_contacts!(agent, model, t)
    dur = dur_I(model.params, agent.age)
    schedule!(agent.id, t + dur, :exit_I, model)
end

function exit_I!(agent::Person, model, t)
    params = model.params
    age    = agent.age
    if rand() <= p_H(params, age)
        agent.status = :H
        dur = dur_H(params, age)
        schedule!(agent.id, t + dur, :exit_H, model)
    else
        agent.status = :R
    end
end

function exit_H!(agent::Person, model, t)
    params = model.params
    age    = agent.age
    if rand() <= p_C(params, age)
        agent.status = :C
        dur = dur_C(params, age)
        schedule!(agent.id, t + dur, :exit_C, model)
    else
        agent.status = :R
    end
end

function exit_C!(agent::Person, model, t)
    params = model.params
    age    = agent.age
    if rand() <= p_V(params, age)
        agent.status = :V
        dur = dur_V(params, age)
        schedule!(agent.id, t + dur, :exit_V, model)
    else
        agent.status = :R
    end
end

function exit_V!(agent::Person, model, t)
    agent.status = rand() <= p_D(model.params, agent.age) ? :D : :R
end

################################################################################
# Functions called by event functions

dur_E(params, age) = rand(Poisson(exp(params.b0_E + params.b1_E * age)))
dur_I(params, age) = rand(Poisson(exp(params.b0_I + params.b1_I * age)))
dur_H(params, age) = rand(Poisson(exp(params.b0_H + params.b1_H * age)))
dur_C(params, age) = rand(Poisson(exp(params.b0_C + params.b1_C * age)))
dur_V(params, age) = rand(Poisson(exp(params.b0_V + params.b1_V * age)))

p_H(params, age) = 1.0 / (1.0 + exp(-(params.a0_H + params.a1_H * age)))  # Pr(Next state is H | age)
p_C(params, age) = 1.0 / (1.0 + exp(-(params.a0_C + params.a1_C * age)))  # Pr(Next state is C | age)
p_V(params, age) = 1.0 / (1.0 + exp(-(params.a0_V + params.a1_V * age)))  # Pr(Next state is V | age)
p_D(params, age) = 1.0 / (1.0 + exp(-(params.a0_D + params.a1_D * age)))  # Pr(Next state is D | age)

p_infect(params, age) = 1.0 / (1.0 + exp(-(params.a0_infect + params.a1_infect * age)))  # Pr(Infect contact | age)

function infect_contacts!(agent::Person, model, t::Int)
    agents    = model.agents
    params    = model.params
    pr_infect = p_infect(params, agent.age)
    for id in agent.household
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
    for id in agent.work
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
    for id in agent.community
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
    for id in agent.social
        infect_contact!(pr_infect, params, agents[id], model, t)
    end
end

"Infect contact with probability p_infect."
function infect_contact!(pr_infect::Float64, params::T, contact::Person, model, t::Int) where {T <: NamedTuple}
    if contact.status == :S
        if rand() <= pr_infect
            contact.status = :E
            dur = dur_E(params, contact.age)
            schedule!(contact.id, t + dur, :exit_E, model)
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