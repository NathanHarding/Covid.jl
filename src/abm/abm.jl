"""
SEIHCVRD age-specific transition probabilities and durations.
The age structure is read from disk.

Pr(next state is j | current state is i, age) = 1 / (1 + exp(-(a0 + a1*age)))

duration|age ~ Poisson(lambda(age)), where lambda(age) = exp(b0 + b1*age)
"""
module abm

export init_output, reset_output!, execute_events!, metrics_to_output!  # Re-exported from the core module

using DataFrames
using Distributions

include(joinpath("..", "core.jl"))
using .core
import .core: unfit!  # To be extended by model-specific method
import .core: fit!    # To be extended by model-specific method

# Model-specific dependencies
include("config.jl")
include("contacts.jl")

using .config
using .contacts

const status0 = Symbol[]  # Used when resetting the model at the beginning of a run
const metrics = Dict(:S => 0, :E => 0, :I => 0, :H => 0, :C => 0, :V => 0, :R => 0, :D => 0)
const active_scenario = Scenario(0.0, 0.0, 0.0, 0.0, 0.0)

function update_active_scenario!(scenario::Scenario)
    active_scenario.household = scenario.household
    active_scenario.school    = scenario.school
    active_scenario.workplace = scenario.workplace
    active_scenario.community = scenario.community
    active_scenario.social    = scenario.social
end

function init_model(indata::Dict{String, DataFrame}, params::T, cfg) where {T <: NamedTuple}
    # Init model
    agedist  = indata["age_distribution"]
    #npeople  = round(Int, 0.1 * sum(agedist.count))  # TODO: Use entire population
    npeople  = round(Int, sum(agedist.count))
    agents   = Vector{Person}(undef, npeople)
    schedule = init_schedule(cfg.maxtime)
    model    = Model(agents, params, 0, cfg.maxtime, schedule)

    # Construct people
    d_age = Categorical(agedist.proportion)
    for id = 1:npeople
        age        = agedist[rand(d_age), :age]
        agents[id] = Person(id, :S, age)
        push!(status0, :S)
    end

    # Seed cases
    rg = 1:npeople
    for (status, n) in cfg.initial_status_counts
        nsuccesses = 0
        jmax = 10 * npeople  # Ceiling on the number of iterations
        for j = 1:jmax
            id = rand(rg)
            agents[id].status != :S && continue  # We've already reset this person's status
            agents[id].status  = status
            status0[id] = status
            nsuccesses += 1
            nsuccesses == n && break
        end
    end

    # Sort agents and populate contacts
    populate_contacts!(agents, params, indata)
    model
end

function reset_model!(model)
    # Empty the schedule
    model.schedule = init_schedule(model.maxtime)

    # Reset each agent's state and schedule a state change
    agents = model.agents
    params = model.params
    n = length(agents)
    for id = 1:n
        agent  = agents[id]
        status = status0[id]
        agent.status = status
        if status == :E
            dur = dur_E(params, agent.age)
            schedule!(id, dur, exit_E!, model)
        elseif status == :I
            dur = dur_I(params, agent.age)
            schedule!(id, dur, exit_I!, model)
        elseif status == :H
            dur = dur_H(params, agent.age)
            schedule!(id, dur, exit_H!, model)
        elseif status == :C
            dur = dur_C(params, agent.age)
            schedule!(id, dur, exit_C!, model)
        elseif status == :V
            dur = dur_V(params, agent.age)
            schedule!(id, dur, exit_V!, model)
        end
    end
end

################################################################################

mutable struct Person <: AbstractAgent
    id::Int
    status::Symbol  # S, E, I, H, C, V, R, D

    # Contacts
    household::Vector{Int}  # People in the same household
    school::Vector{Int}     # Child care, primary school, secondary school, university. Contains both teachers and students.
    workplace::Vector{Int}  # Workplace. Empty for teachers, whose workplace is school.
    community::Vector{Int}  # Shops, transport, pool, library, etc
    social::Vector{Int}     # Family and/or friends outside the household

    # Risk factors
    age::Int
end

Person(id::Int, status::Symbol, age::Int) = Person(id, status, Int[], Int[], Int[], Int[], Int[], age)

function exit_S!(agent::Person, model, t)
    agent.status = :E
    dur = dur_E(model.params, agent.age)
    schedule!(agent.id, t + dur, exit_E!, model)
end

function exit_E!(agent::Person, model, t)
    agent.status = :I
    dur = dur_I(model.params, agent.age)
    schedule!(agent.id, t + dur, exit_I!, model)
    infect_contacts!(agent, model, t)  # Schedules an immediate status change for each contact
end

function exit_I!(agent::Person, model, t)
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

function exit_H!(agent::Person, model, t)
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

function exit_C!(agent::Person, model, t)
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

function exit_V!(agent::Person, model, t)
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

function infect_contacts!(agent::Person, model, t::Int)
    agent.status != :I && return
    agents      = model.agents
    age         = agent.age
    pr_infect   = p_infect(model.params, age)
    n_susceptible_contacts = 0
    n_susceptible_contacts = infect_contacts!(:household, active_scenario.household, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_contacts!(:school,    active_scenario.school,    pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_contacts!(:workplace, active_scenario.workplace, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_contacts!(:community, active_scenario.community, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_contacts!(:social,    active_scenario.social,    pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts > 0 && schedule!(agent.id, t + 1, infect_contacts!, model)
end

function infect_contacts!(contact_category::Symbol, pr_contact, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    contactlist = getproperty(agent, contact_category)
    for id in contactlist
        contact = agents[id]
        if pr_contact > 0.0 && rand() <= pr_contact         # Contact between agent and contact occurs
            if contact.status == :S && rand() <= pr_infect  # The agent infects the contact
                execute_event!(exit_S!, contact, model, t, metrics)
            #elseif contact.status == :R && rand() <= contact.p_reinfection
            #    execute_event!(exit_S!, contact, model, t, metrics)
            end
        end
        if contact.status == :S
            n_susceptible_contacts += 1
        end
    end
    n_susceptible_contacts
end

################################################################################
# Metrics

function reset_metrics!(model)
    for (k, v) in metrics
        metrics[k] = 0
    end
    for agent in model.agents
        metrics[agent.status] += 1
    end
end

"Remove the agent's old state from metrics"
function unfit!(metrics, agent::Person)
    metrics[agent.status] -= 1
end

"Add the agent's new state to metrics"
function fit!(metrics, agent::Person)
    metrics[agent.status] += 1
end

end