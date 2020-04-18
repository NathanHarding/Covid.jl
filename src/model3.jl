"""
SEIHCVRD with Conductor
"""
module model3

using Agents

################################################################################
# Infrastructure (not specific to this model)

abstract type AbstractEventBag end  # An instance is a set of events that occur simultaneously

"A conductor keeps time and executes scheduled events"
mutable struct Conductor{T <: AbstractEventBag}
    time::Int
    schedule0::T  # Events that take place during (0, 1)
    schedule::Vector{T}
end

Conductor(nsteps::Int, T::DataType) = Conductor(0, T(), [T() for i = 1:nsteps])

"Schedule an event at time t for agent id"
function schedule!(agent_id, t, event::Symbol, model)
    func, ids = getfield(model.properties.schedule[t], event)
    push!(ids, agent_id)
end

"Execute an event bag"
function execute!(eb::T, t::Int, model) where {T <: AbstractEventBag}
    agents = model.agents
    for fname in fieldnames(typeof(eb))
        func, ids = getfield(eb, fname)
        for id in ids
            func(agents[id], model, t)
        end
    end
end

scheduler(model) = ""  # Any object for which isempty(x) is true

agent_step!(agent, model) = nothing

function model_step!(model)
    conductor = model.properties
    t  = conductor.time
    eb = t == 0 ? conductor.schedule0 : conductor.schedule[t]
    execute!(eb, t, model)
    conductor.time += 1
end

function init_model(griddims::Tuple{Int, Int}, npeople::Int, prms::Dict{Symbol, Any}, dist0, conductor)
    space = Space(griddims, moore=true)
    model = AgentBasedModel(Person, space; scheduler=scheduler, properties=conductor)
    cdf0  = cumsum(dist0)
    i     = griddims[1]
    j     = griddims[2]
    for id in 1:npeople
        j = j == griddims[2] ? 1 : j + 1
        i = j == 1 ? i + 1 : i
        i = i > griddims[1] ? 1 : i
        person = Person(id, (i, j), prms, cdf0, model)
        add_agent!(person, person.pos, model)
    end
    model
end

################################################################################
# The model

mutable struct Person <: AbstractAgent
    id::Int
    pos::Tuple{Int, Int}
    status::Symbol  # S, E, I, H, C, V, R, D

    ### Constants (determined when person is created prior to execution)
    # Random variables
    dur_exposed::Int       # Number of days the person is in state E (infected but not contagious)
    dur_infectious::Int    # Number of days the person is in state I (infected and contagious)
    dur_hospitalised::Int  # Number of days the person is in state H
    dur_icu::Int           # Number of days the person is in state C (Care)
    dur_ventilated::Int    # Number of days the person is in state V 
    
    # Probabilities
    p_infect::Float64       # Pr(Infecting a Susceptible person | Status is Infectious and Contact with Susceptible person)
    p_hospital::Float64     # Pr(Hospitalisation | Infectious). Pr(Recovery | Infectious) = 1 - Pr(Hospitalisation | Infectious).
    p_icu::Float64          # Pr(ICU | Hospitalisation). Pr(Recovery | Hospitalisation) = 1 - Pr(ICU | Hospitalisation).
    p_ventilation::Float64  # Pr(Ventilation | ICU). Pr(Recovery | ICU) = 1 - Pr(Ventilation | ICU).
    p_death::Float64        # Pr(Death | Ventilation).  Pr(Recovery | Ventilation) = 1 - Pr(Death | Ventilation).
    p_reinfection           # Pr(Re-infection | Recovered and Contact with Infectious person)

    # Risk factors
    age::Int
end

function Person(id::Int, pos, prms::Dict{Symbol, Any}, cdf0::Vector{Float64}, model)
    r = rand()
    if r <= cdf0[1]
        status = :S
    elseif r <= cdf0[2]
        status = :E
        schedule!(id, prms[:dur_exposed], :exit_E, model)
    elseif r <= cdf0[3]
        status = :I
        schedule!(id, prms[:dur_infectious], :exit_I, model)
    elseif r <= cdf0[4]
        status = :H
        schedule!(id, prms[:dur_hospitalised], :exit_H, model)
    elseif r <= cdf0[5]
        status = :C
        schedule!(id, prms[:dur_icu], :exit_C, model)
    elseif r <= cdf0[6]
        status = :V
        schedule!(id, prms[:dur_ventilated], :exit_V, model)
    elseif r <= cdf0[7]
        status = :R
    else
        status = :D
    end
    Person(id, pos, status,
           prms[:dur_exposed], prms[:dur_infectious], prms[:dur_hospitalised], prms[:dur_icu], prms[:dur_ventilated],
           prms[:p_infect], prms[:p_hospital], prms[:p_icu], prms[:p_ventilation], prms[:p_death], prms[:p_reinfection],
           prms[:age])
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
    schedule!(agent.id, t + agent.dur_infectious, :exit_I, model)
end

function exit_I!(agent::Person, model, t)
    if rand() <= agent.p_hospital
        agent.status = :H
        schedule!(agent.id, t + agent.dur_hospitalised, :exit_H, model)
    else
        agent.status = :R
    end
end

function exit_H!(agent::Person, model, t)
    if rand() <= agent.p_icu
        agent.status = :C
        schedule!(agent.id, t + agent.dur_icu, :exit_C, model)
    else
        agent.status = :R
    end
end

function exit_C!(agent::Person, model, t)
    if rand() <= agent.p_ventilation
        agent.status = :V
        schedule!(agent.id, t + agent.dur_ventilated, :exit_V, model)
    else
        agent.status = :R
    end
end

function exit_V!(agent::Person, model, t)
    agent.status = rand() <= agent.p_death ? :D : :R
end

################################################################################
# Functions called by event functions

function infect_contacts!(agent::Person, model, t::Int)
    # Infect Susceptible contacts in the same node
    p_infect = agent.p_infect
    agents   = model.agents
    same_node_ids = get_node_contents(agent, model)  # Includes agent.id
    for same_node_id in same_node_ids
        same_node_id == agent.id && continue  # Agent cannot infect him/herself
        infect_contact!(agents[same_node_id], model, t, p_infect)
    end

    # Infect Susceptible contacts in neighbouring nodes
    agent_node = coord2vertex(agent.pos, model)
    neighbor_coords = Agents.neighbors(model.space.graph, agent_node)
    for neighbor_coord in neighbor_coords
        neighbour_ids = get_node_contents(neighbor_coord, model)
        for neighbour_id in neighbour_ids
            infect_contact!(agents[neighbour_id], model, t, p_infect)
        end
    end
end

"Infect contact with probability p_infect."
function infect_contact!(contact::Person, model, t::Int, p_infect::Float64)
    if contact.status == :S
        if rand() <= p_infect
            contact.status = :E
            schedule!(contact.id, t + contact.dur_exposed, :exit_E, model)
        end
    elseif contact.status == :R
        if rand() <= contact.p_reinfection
            contact.status = :E
            schedule!(contact.id, t + contact.dur_exposed, :exit_E, model)
        end
    end
end

end