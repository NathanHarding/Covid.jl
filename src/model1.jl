module model1

using Agents

mutable struct Person <: AbstractAgent
    id::Int
    pos::Int
    status::Char  # S, E, I, R, D

    # Random variables    
    dur_exposed::Int         # Number of days the person is infected but not contagious
    dur_infectious::Int      # Number of days the person is infected and contagious

    # Probabilities
    infectiousness::Float64  # Pr(Infecting a Susceptible person | Contact with Susceptible person)
    p_death::Float64         # Pr(Death | Infected)
    p_reinfection            # Pr(Infectable | Recovered)

    # Risk factors
    age::Int
end

function init_agent(id::Int, pos, prms::Dict{Symbol, Any})
    Person(id, pos, prms[:dur_exposed], prms[:dur_infectious], prms[:infectiousness], prms[:p_death], prms[:p_reinfection], prms[:age])
end

function init_model(griddims::Tuple{Int, Int}, npeople::Int, prms::Dict{Symbol, Any}; scheduler=fastest, properties=nothing)
    space = Space(griddims, moore=true)
    model = AgentBasedModel(Person, space; scheduler, properties)
    for i in 1:npeople
        person = init_person(i, (1,1), prms)
        add_agent_single!(person, model)
    end
    model
end

function agent_step!(agent, model)
end

function run_model(griddims::Tuple{Int, Int}, npeople::Int, prms::Dict{Symbol, Any}; nsteps::Int=1, scheduler=fastest, properties=nothing)
    model = init_model(griddims, npeople, prms; scheduler=scheduler, properties=properties)
    data  = step!(model, agent_step!, nsteps, props, when=when)
end

end