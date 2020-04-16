module model1

using Agents

mutable struct Person <: AbstractAgent
    id::Int
    pos::Tuple{Int, Int}
    status::Char  # S, E, I, R, D

    ### Constants (determined when person is created prior to execution)
    # Random variables
    dur_exposed::Int         # Number of days the person is infected but not contagious
    dur_infectious::Int      # Number of days the person is infected and contagious

    # Probabilities
    p_infect::Float64  # Pr(Infecting a Susceptible person | Status is Infectious and Contact with Susceptible person)
    p_death::Float64   # Pr(Death | Infected)
    p_reinfection      # Pr(Re-infeection | Recovered and Contact with Infectious person)

    ### Conveniences for faster execution
    # Time of status exit
    t_exit_exposed::Int      # Time that person exits Exposed status. Populated when person enters Exposed status.
    t_exit_infectious::Int   # Time that person exits Infectious status. Populated when person enters Infectious status.
end

function init_person(id::Int, pos, prms::Dict{Symbol, Any}, cdf0::Vector{Float64})
    r = rand()
    t_exit_exposed    = 0
    t_exit_infectious = 0
    if r <= cdf0[1]
        status = 'S'
    elseif r <= cdf0[2]
        status = 'E'
        t_exit_exposed = prms[:dur_exposed]
    elseif r <= cdf0[3]
        status = 'I'
        t_exit_infectious = prms[:dur_infectious]
    elseif r <= cdf0[4]
        status = 'R'
    else
        status = 'D'
    end
    Person(id, pos, status, prms[:dur_exposed], prms[:dur_infectious], prms[:p_infect], prms[:p_death], prms[:p_reinfection], t_exit_exposed, t_exit_infectious)
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
    # Get required info
    t = model.properties[:time]
    status = agent.status

    # Transition from Exposed
    if status == 'E' && agent.t_exit_exposed == t
        agent.status = 'I'
        agent.t_exit_infectious = t + agent.dur_infectious
        return
    end

    # Nothing to do if status is S, E-ongoing, R or D
    status != 'I' && return

    # Transition from Infectious
    if agent.t_exit_infectious == t
        agent.status = rand() <= agent.p_death ? 'D' : 'R'
        return
    end

    # Infect Susceptible contacts in the same node
    p_infect = agent.p_infect
    same_node_ids = get_node_contents(agent, model)  # Includes agent.id
    for same_node_id in same_node_ids
        same_node_id == agent.id && continue  # Agent cannot infect him/herself
        infect!(t, model.agents[same_node_id], p_infect)
    end

    # Infect Susceptible contacts in neighbouring nodes
    neighbor_coords = node_neighbors(agent, model)  # Excludes agent.pos
    for neighbor_coord in neighbor_coords
        neighbour_ids = get_node_contents(neighbor_coord, model)
        for neighbour_id in neighbour_ids
            infect!(t, model.agents[neighbour_id], p_infect)
        end
    end
end

"Infect contact with probability p_infect."
function infect!(t::Int, contact::Person, p_infect::Float64)
    r = rand()
    if (contact.status == 'S' && r <= p_infect) || (contact.status == 'R' && r <= contact.p_reinfection)
        contact.status = 'E'
        contact.t_exit_exposed = t + contact.dur_exposed
    end
end

end