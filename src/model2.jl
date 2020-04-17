module model2

using Agents

mutable struct Person <: AbstractAgent
    id::Int
    pos::Tuple{Int, Int}
    status::Char  # S, E, I, H, C, V, R, D

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
    p_reinfection           # Pr(Re-infeection | Recovered and Contact with Infectious person)

    # Risk factors
    age::Int

    ### Conveniences for faster execution
    # Time of status exit
    t_exit_exposed::Int       # Time that person exits Exposed status. Populated when person enters Exposed status.
    t_exit_infectious::Int    # Time that person exits Infectious status. Populated when person enters Infectious status.
    t_exit_hospitalised::Int  # Time that person exits Hospitalised status. Populated when person enters Hospitalised status.
    t_exit_icu::Int           # Time that person exits ICU (C) status. Populated when person enters ICU status.
    t_exit_ventilated::Int    # Time that person exits Ventilated status. Populated when person enters Ventilated status.
end

function init_person(id::Int, pos, prms::Dict{Symbol, Any}, cdf0::Vector{Float64})
    r = rand()
    t_exit_exposed      = 0
    t_exit_infectious   = 0
    t_exit_hospitalised = 0
    t_exit_icu          = 0
    t_exit_ventilated   = 0
    if r <= cdf0[1]
        status = 'S'
    elseif r <= cdf0[2]
        status = 'E'
        t_exit_exposed = prms[:dur_exposed]
    elseif r <= cdf0[3]
        status = 'I'
        t_exit_infectious = prms[:dur_infectious]
    elseif r <= cdf0[4]
        status = 'H'
        t_exit_hospitalised = prms[:dur_hospitalised]
    elseif r <= cdf0[5]
        status = 'C'
        t_exit_icu = prms[:dur_icu]
    elseif r <= cdf0[6]
        status = 'V'
        t_exit_ventilated = prms[:dur_ventilated]
    elseif r <= cdf0[7]
        status = 'R'
    else
        status = 'D'
    end
    Person(id, pos, status,
           prms[:dur_exposed], prms[:dur_infectious], prms[:dur_hospitalised], prms[:dur_icu], prms[:dur_ventilated],
           prms[:p_infect], prms[:p_hospital], prms[:p_icu], prms[:p_ventilation], prms[:p_death], prms[:p_reinfection],
           prms[:age],
           t_exit_exposed, t_exit_infectious, t_exit_hospitalised, t_exit_icu, t_exit_ventilated)
end

function init_model(griddims::Tuple{Int, Int}, npeople::Int, prms::Dict{Symbol, Any}, dist0; scheduler=fastest, properties=nothing)
    space = Space(griddims, moore=true)
    model = AgentBasedModel(Person, space; scheduler=scheduler, properties=properties)
    cdf0  = cumsum(dist0)
    i     = griddims[1]
    j     = griddims[2]
    for id in 1:npeople
        j = j == griddims[2] ? 1 : j + 1
        i = j == 1 ? i + 1 : i
        i = i > griddims[1] ? 1 : i
        person = init_person(id, (i, j), prms, cdf0)
        add_agent!(person, person.pos, model)
    end
    model
end

function agent_step!(agent, model)
    t = model.properties[:time]
    status = agent.status
    if status == 'E' && agent.t_exit_exposed == t           # Transition from Exposed
        agent.status = 'I'
        agent.t_exit_infectious = t + agent.dur_infectious
    elseif status == 'I'
        if agent.t_exit_infectious == t                     # Transition from Infectious
            if rand() <= agent.p_hospital
                agent.status = 'H'
                agent.t_exit_hospitalised = t + agent.dur_hospitalised
            else
                agent.status = 'R'
            end
        else
            infect_contacts!(agent, model, t)               # Infect contacts
        end
    elseif status == 'H' && agent.t_exit_hospitalised == t  # Transition from Hospitalised
        if rand() <= agent.p_icu
            agent.status = 'C'
            agent.t_exit_icu = t + agent.dur_icu
        else
            agent.status = 'R'
        end
    elseif status == 'C' && agent.t_exit_icu == t           # Transition from ICU
        if rand() <= agent.p_ventilation
            agent.status = 'V'
            agent.t_exit_ventilated = t + agent.dur_ventilated
        else
            agent.status = 'R'
        end
    elseif status == 'V' && agent.t_exit_ventilated == t    # Transition from Ventilated
        agent.status = rand() <= agent.p_death ? 'D' : 'R'
    end
end

function infect_contacts!(agent::Person, model, t::Int)
    # Infect Susceptible contacts in the same node
    p_infect = agent.p_infect
    same_node_ids = get_node_contents(agent, model)  # Includes agent.id
    for same_node_id in same_node_ids
        same_node_id == agent.id && continue  # Agent cannot infect him/herself
        infect_contact!(t, model.agents[same_node_id], p_infect)
    end

    # Infect Susceptible contacts in neighbouring nodes
    agent_node = coord2vertex(agent.pos, model)
    neighbor_coords = Agents.neighbors(model.space.graph, agent_node)
    for neighbor_coord in neighbor_coords
        neighbour_ids = get_node_contents(neighbor_coord, model)
        for neighbour_id in neighbour_ids
            infect_contact!(t, model.agents[neighbour_id], p_infect)
        end
    end
end

"Infect contact with probability p_infect."
function infect_contact!(t::Int, contact::Person, p_infect::Float64)
    r = rand()
    if (contact.status == 'S' && r <= p_infect) || (contact.status == 'R' && r <= contact.p_reinfection)
        contact.status = 'E'
        contact.t_exit_exposed = t + contact.dur_exposed
    end
end

end