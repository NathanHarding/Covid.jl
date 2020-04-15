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

    # Risk factors
    age::Int

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
    Person(id, pos, status, prms[:dur_exposed], prms[:dur_infectious], prms[:p_infect], prms[:p_death], prms[:p_reinfection], prms[:age], t_exit_exposed, 0)
end

function init_model(griddims::Tuple{Int, Int}, npeople::Int, prms::Dict{Symbol, Any}, dist0; scheduler=fastest, properties=nothing)
    space = Space(griddims, moore=true)
    model = AgentBasedModel(Person, space; scheduler=scheduler, properties=properties)
    cdf0  = cumsum(dist0)
    for i in 1:npeople
        person = init_person(i, (1,1), prms, cdf0)
        add_agent_single!(person, model)
    end
    model
end

function agent_step!(agent, model)
    t = model.properties[:time]
    status = agent.status
    if status == 'E' && agent.t_exit_exposed == t  # Transition from Exposed
        agent.status == 'I'
        agent.t_exit_infectious = t + agent.dur_infectious
    elseif status == 'I'
        if agent.t_exit_infectious == t  # Transition from Infectious
            agent.status = rand() <= agent.p_death ? 'D' : 'R'
        else  # Infect Susceptible neighbours
            p_infect       = agent.p_infect
            neighbor_cells = node_neighbors(agent, model)
            for neighbor_cell in neighbor_cells
                neighbour_ids = get_node_contents(neighbor_cell, model)
                for neighbour_id in neighbour_ids
                    neighbour = model[neighbour_id]
                    r = rand()
                    if neighbour.status == 'S' && r <= p_infect
                        neighbour.status = 'I'
                    elseif neighbour.status == 'R' && r <= neighbour.p_reinfection
                        neighbour.status = 'I'
                    end
                end
            end
        end
    end
end

end