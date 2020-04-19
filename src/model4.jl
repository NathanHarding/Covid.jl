"""
SEIHCVRD without Agents.jl, and with Distributions.
"""
module model4

using DataFrames
using Distributions

################################################################################
# Infrastructure (not specific to this model)

abstract type AbstractAgent end
abstract type AbstractEventBag end  # An instance is a set of events that occur simultaneously

mutable struct Model{A <: AbstractAgent, E <: AbstractEventBag, T <: NamedTuple}
    agents::Vector{A}
    params::T
    time::Int
    maxtime::Int
    schedule0::E         # Events that take place during (0, 1)
    schedule::Vector{E}  # Events that take place during (1, maxtime)
end

"Schedule an event at time t for agent id"
function schedule!(agent_id, t, event::Symbol, model)
    eb = t == 0 ? model.schedule0 : model.schedule[t]
    func, ids = getfield(eb, event)
    push!(ids, agent_id)
end

"Execute an event bag"
function execute!(eb::T, model, t::Int) where {T <: AbstractEventBag}
    agents = model.agents
    for fname in fieldnames(typeof(eb))
        func, ids = getfield(eb, fname)
        for id in ids
            func(agents[id], model, t)
        end
    end
end

function run!(model)
    output, status2rownum = init_output(model)
    collectdata!(model, output, 0, status2rownum)  # t == 0
    execute!(model.schedule0, model, 0)  # t in (0, 1)
    for t = 1:(model.maxtime - 1)
        model.time = t
        collectdata!(model, output, t, status2rownum)
        execute!(model.schedule[t], model, t)
    end
    collectdata!(model, output, model.maxtime, status2rownum)
    output_to_dataframe(output, status2rownum)
end

function init_output(model)
    output = fill(0.0, 8, model.maxtime + 1)  # output[:, t] is the observation (SEIHCVRD) at time t
    status2rownum = Dict(:S => 1, :E => 2, :I => 3, :H => 4, :C => 5, :V => 6, :R => 7, :D => 8)
    output, status2rownum
end

function collectdata!(model, output, t, status2rownum)
    j = t + 1
    agents = model.agents
    for agent in agents
        output[status2rownum[agent.status], j] += 1
    end
end

function output_to_dataframe(output, status2rownum)
    ni = size(output, 2)
    nj = size(output, 1)
    j2name   = Dict(v => k for (k, v) in status2rownum)
    colnames = vcat(:time, [j2name[j] for j = 1:nj])
    result   = DataFrame(fill(Int, length(colnames)), colnames, ni)
    for i = 1:ni
        result[i, :time] = i - 1
        for (j, colname) in j2name
            result[i, colname] = output[j, i]
        end
    end
    result
end

################################################################################
# The model

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