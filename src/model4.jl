"""
SEIHCVRD without Agents.jl, and with Distributions.
"""
module model4

using DataFrames
using Distributions

using ..core

function init_model(npeople::Int, params::T, maxtime, dist0) where {T <: NamedTuple}
    agents    = Vector{Person}(undef, npeople)
    schedule0 =  EventBag()
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
    params = model.params
    r = rand()
    if r <= cdf0[1]
        status = :S
    elseif r <= cdf0[2]
        status = :E
        dur_E  = rand(Poisson(params.mean_dur_E))
        schedule!(id, dur_E, :exit_E, model)
    elseif r <= cdf0[3]
        status = :I
        dur_I  = rand(Poisson(params.mean_dur_I))
        schedule!(id, dur_I, :exit_I, model)
    elseif r <= cdf0[4]
        status = :H
        dur_H  = rand(Poisson(params.mean_dur_H))
        schedule!(id, dur_H, :exit_H, model)
    elseif r <= cdf0[5]
        status = :C
        dur_C  = rand(Poisson(params.mean_dur_C))
        schedule!(id, dur_C, :exit_C, model)
    elseif r <= cdf0[6]
        status = :V
        dur_V  = rand(Poisson(params.mean_dur_V))
        schedule!(id, dur_V, :exit_V, model)
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
    Person(id, status, household, work, community, social, 20)
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
    dur_I = rand(Poisson(model.params.mean_dur_I))
    schedule!(agent.id, t + dur_I, :exit_I, model)
end

function exit_I!(agent::Person, model, t)
    params = model.params
    if rand() <= params.p_H
        agent.status = :H
        dur_H = rand(Poisson(params.mean_dur_H))
        schedule!(agent.id, t + dur_H, :exit_H, model)
    else
        agent.status = :R
    end
end

function exit_H!(agent::Person, model, t)
    params = model.params
    if rand() <= params.p_C
        agent.status = :C
        dur_C = rand(Poisson(params.mean_dur_C))
        schedule!(agent.id, t + dur_C, :exit_C, model)
    else
        agent.status = :R
    end
end

function exit_C!(agent::Person, model, t)
    params = model.params
    if rand() <= params.p_V
        agent.status = :V
        dur_V = rand(Poisson(params.mean_dur_V))
        schedule!(agent.id, t + dur_V, :exit_V, model)
    else
        agent.status = :R
    end
end

function exit_V!(agent::Person, model, t)
    agent.status = rand() <= model.params.p_death ? :D : :R
end

################################################################################
# Functions called by event functions

function infect_contacts!(agent::Person, model, t::Int)
    agents     = model.agents
    p_infect   = model.params.p_infect
    mean_dur_E = model.params.mean_dur_E
    for id in agent.household
        infect_contact!(p_infect, mean_dur_E, agents[id], model, t)
    end
    for id in agent.work
        infect_contact!(p_infect, mean_dur_E, agents[id], model, t)
    end
    for id in agent.community
        infect_contact!(p_infect, mean_dur_E, agents[id], model, t)
    end
    for id in agent.social
        infect_contact!(p_infect, mean_dur_E, agents[id], model, t)
    end
end

"Infect contact with probability p_infect."
function infect_contact!(p_infect::Float64, mean_dur_E::Float64, contact::Person, model, t::Int)
    if contact.status == :S
        if rand() <= p_infect
            contact.status = :E
            dur_E = rand(Poisson(mean_dur_E))
            schedule!(contact.id, t + dur_E, :exit_E, model)
        end
    end
    #=
    elseif contact.status == :R
        if rand() <= contact.p_reinfection
            contact.status = :E
            dur_E = rand(Poisson(mean_dur_E))
            schedule!(contact.id, t + dur_E, :exit_E, model)
        end
    end
    =#
end

end