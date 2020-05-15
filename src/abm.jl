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

include("core.jl")
using .core
import .core: unfit!  # To be extended by model-specific method
import .core: fit!    # To be extended by model-specific method

# Model-specific dependencies
include("config.jl")
include("contacts.jl")

using .config
using .contacts

const metrics = Dict(:S => 0, :E => 0, :I1 => 0, :I2 => 0, :H => 0, :C => 0, :V => 0, :R => 0, :D => 0, :positives => 0)
const active_distancing_regime = DistancingRegime(0.0, 0.0, 0.0, 0.0, 0.0)

# Conveniences
const status0    = Symbol[]       # Used when resetting the model at the beginning of a run
const households = Household[]    # households[i].adults[j] is the id of the jth adult in the ith household. Ditto children.
const workplaces = Vector{Int}[]  # workplaces[i][j] is the id of the jth worker in the ith workplace.
const communitycontacts = Int[]   # Contains person IDs. Social contacts can be derived for each person.
const socialcontacts    = Int[]

################################################################################

function update_active_distancing_regime!(distancing_regime::DistancingRegime)
    active_distancing_regime.household = distancing_regime.household
    active_distancing_regime.school    = distancing_regime.school
    active_distancing_regime.workplace = distancing_regime.workplace
    active_distancing_regime.community = distancing_regime.community
    active_distancing_regime.social    = distancing_regime.social
end

mutable struct Person <: AbstractAgent
    id::Int
    status::Symbol  # S, E, I1, I2, I3, H, C, V, R, D
    tested::Bool
    has_been_to_icu::Bool
    has_been_ventilated::Bool

    # Risk factors
    age::Int

    # Contacts
    i_household::Int  # Person is in households[i_household].
    school::Union{Nothing, Vector{Int}}  # Child care, primary school, secondary school, university. Not empty for teachers and students.
    ij_workplace::Union{Nothing, Tuple{Int, Int}}  # Empty for children and teachers (whose workplace is school). person.id == workplaces[i][j].
    i_community::Int  # Shops, transport, pool, library, etc. communitycontacts[i_community] == person.id
    i_social::Int     # Family and/or friends outside the household. socialcontacts[i_social] == person.id
end

Person(id::Int, status::Symbol, age::Int) = Person(id, status, false, false, false, age, 0, nothing, nothing, 0, 0)

function init_model(indata::Dict{String, DataFrame}, params::T, cfg) where {T <: NamedTuple}
    # Init model
    agedist  = indata["age_distribution"]
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
    populate_contacts!(agents, params, indata, households, workplaces, communitycontacts, socialcontacts)
    model
end

function reset_model!(model)
    model.schedule = init_schedule(model.maxtime)  # Empty the schedule
    agents = model.agents
    params = model.params
    for agent in agents  # Reset each agent's state and schedule a state change
        agent.tested = false
        status = status0[agent.id]
        if status == :E
            to_E!(agent, model, 0)
        elseif status == :I1
            to_I1!(agent, model, 0)
        elseif status == :I2
            to_I2!(agent, model, 0)
        elseif status == :H
            to_H!(agent, model, 0)
        elseif status == :C
            to_C!(agent, model, 0)
        elseif status == :V
            to_V!(agent, model, 0)
        elseif status == :R
            to_R!(agent, model, 0)
        elseif status == :D
            to_D!(agent, model, 0)
        end
    end
end

################################################################################

function test_for_covid!(agent::Person, model, t)
    agent.tested = true
    metrics[:positives] += 1
    schedule!(agent.id, t + 2, get_test_result!, model)  # Test result available 2 days after test
end

function get_test_result!(agent::Person, model, t)
end

function to_E!(agent::Person, model, t)
    agent.status = :E
    dur = dur_E(model.params, agent.age)
    schedule!(agent.id, t + dur, to_I1!, model)
end

function to_I1!(agent::Person, model, t)
    agent.status = :I1
    params = model.params
    age    = agent.age
    if rand() <= p_I2(params, age)  # Person will progress from I1 to I2
        dur = dur_I1(params, age)
        schedule!(agent.id, t + dur, to_I2!, model)
    else  # Person will progress from I1 to Recovered
        dur = dur_I1(params, age) + dur_I2(params, age)  # Assume asymptomatic duration is the same as pre-hospitalisation duration
        schedule!(agent.id, t + dur, to_R!, model)
    end
    infect_contacts!(agent, model, t)  # Schedules an immediate status change for each contact
end

function to_I2!(agent::Person, model, t)
    agent.status = :I2
    params  = model.params
    rand() <= params.p_test && schedule!(agent.id, t + 2, test_for_covid!, model)  # Testing in the community and/or Emergency department
    age = agent.age
    dur = dur_I2(params, age)
    if rand() <= p_H(params, age)  # Person will progress from I2 to H
        schedule!(agent.id, t + dur, to_H!, model)
    else  # Person will progress from I2 to Recovered
        schedule!(agent.id, t + dur, to_R!, model)
    end
end

function to_H!(agent::Person, model, t)
    agent.status = :H
    agent.tested == false && test_for_covid!(agent, model, t)  # Agent tests positive when admitted to hospital
    params = model.params
    age    = agent.age
    dur    = dur_H(params, age)
    if agent.has_been_to_icu  # Person is improving after ICU - s/he will progress from the ward to recovery after 3 days
        schedule!(agent.id, t + 3, to_R!, model)
    elseif rand() <= p_C(params, age)  # Person is deteriorating - s/he will progress from H to ICU
        schedule!(agent.id, t + dur, to_C!, model)
    else  # Person will progress from H to Recovered without goiing to ICU
        schedule!(agent.id, t + dur, to_R!, model)
    end
end

function to_C!(agent::Person, model, t)
    agent.status = :C
    agent.has_been_to_icu = true
    params = model.params
    age    = agent.age
    dur    = dur_C(params, age)
    if agent.has_been_ventilated  # Person is improving after ventilation - s/he will progress from ICU to the ward after 3 days
        schedule!(agent.id, t + 3, to_H!, model)
    elseif rand() <= p_V(params, age)  # Person is deteriorating - s/he will progress from a non-ventilated ICU bed to a ventilated ICU bed
        schedule!(agent.id, t + dur, to_V!, model)
    else  # Person will progress from ICU to the ward without going to a ventilated ICU bed
        schedule!(agent.id, t + dur, to_H!, model)
    end
end

function to_V!(agent::Person, model, t)
    agent.status = :V
    agent.has_been_ventilated = true
    params = model.params
    age    = agent.age
    dur    = dur_C(params, age)
    if rand() <= p_D(params, age)  # Person will progress from ventilation to deceased
        schedule!(agent.id, t + dur, to_D!, model)
    else  # Person will progress from ventilation to a non-ventilated ICU bed
        schedule!(agent.id, t + dur, to_C!, model)
    end
end

to_R!(agent::Person, model, t) = agent.status = :R
to_D!(agent::Person, model, t) = agent.status = :D

################################################################################
# Functions called by event functions

dur_E(params, age)  = rand(Poisson(exp(params.b0_E  + params.b1_E  * age)))
dur_H(params, age)  = rand(Poisson(exp(params.b0_H  + params.b1_H  * age)))
dur_I1(params, age) = rand(Poisson(exp(params.b0_I1 + params.b1_I1 * age)))
dur_I2(params, age) = rand(Poisson(exp(params.b0_I2 + params.b1_I2 * age)))
dur_C(params, age)  = rand(Poisson(exp(params.b0_C  + params.b1_C  * age)))
dur_V(params, age)  = rand(Poisson(exp(params.b0_V  + params.b1_V  * age)))

"Pr(Next state is H | Current state is I2, age)."
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

p_I2(params, age) = 1.0 / (1.0 + exp(-(params.a0_I2 + params.a1_I2 * age)))  # Pr(Next state is I2 | Current state is I1, age)
p_C(params, age)  = 1.0 / (1.0 + exp(-(params.a0_C  + params.a1_C  * age)))  # Pr(Next state is C  | Current state is H,  age)
p_V(params, age)  = 1.0 / (1.0 + exp(-(params.a0_V  + params.a1_V  * age)))  # Pr(Next state is V  | Current state is C,  age)
p_D(params, age)  = 1.0 / (1.0 + exp(-(params.a0_D  + params.a1_D  * age)))  # Pr(Next state is D  | Current state is V,  age)

p_infect(params, age) = 1.0 / (1.0 + exp(-(params.a0_infect + params.a1_infect * age)))  # Pr(Infect contact | age)

function infect_contacts!(agent::Person, model, t::Int)
    agent.status != :I1 && agent.status != :I2 && return
    agents    = model.agents
    params    = model.params
    pr_infect = p_infect(params, agent.age)
    n_susceptible_contacts = 0
    n_susceptible_contacts = infect_household_contacts!(households, active_distancing_regime.household, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    if !isnothing(agent.school)
        n_susceptible_contacts = infect_contactlist!(agent.school, active_distancing_regime.school, pr_infect, agents, model, t, n_susceptible_contacts)
    end
    if !isnothing(agent.ij_workplace)
        i, j = agent.ij_workplace
        n_susceptible_contacts = infect_community_contacts!(workplaces[i], j, Int(params.n_workplace_contacts), active_distancing_regime.workplace,
                                                            pr_infect, agent, agents, model, t, n_susceptible_contacts)
    end
    n_susceptible_contacts = infect_community_contacts!(communitycontacts, agent.i_community, Int(params.n_community_contacts), active_distancing_regime.community,
                                                        pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_community_contacts!(socialcontacts, agent.i_social, Int(params.n_social_contacts), active_distancing_regime.social, 
                                                        pr_infect, agent, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts > 0 && schedule!(agent.id, t + 1, infect_contacts!, model)
end

function infect_household_contacts!(households, pr_contact, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    agentid   = agent.id
    household = households[agent.i_household]
    for id in household.adults
        id == agentid && continue
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
    end
    for id in household.children
        id == agentid && continue
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
    end
    n_susceptible_contacts
end

function infect_community_contacts!(community::Vector{Int}, i_agent::Int, ncontacts_per_person::Int,
                                    pr_contact, pr_infect, agent, agents, model, t, n_susceptible_contacts)
    i_agent == 0 && return n_susceptible_contacts
    agentid = agent.id
    npeople = length(community)
    ncontacts_per_person = min(npeople - 1, ncontacts_per_person)
    halfn   = div(ncontacts_per_person, 2)
    i1 = rem(i_agent - halfn + npeople, npeople)
    i1 = i1 == 0 ? npeople : i1
    i2 = rem(i_agent + halfn, npeople)
    i2 = i2 == 0 ? npeople : i2
    if i1 < i2
        for i = i1:i2
            i == i_agent && continue
            id = community[i]
            n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
        end
    elseif i1 > i2
        for i = i1:npeople
            i == i_agent && continue
            id = community[i]
            n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
        end
        for i = 1:i2
            i == i_agent && continue
            id = community[i]
            n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
        end
    end
    if isodd(ncontacts_per_person)
        i  = rem(i_agent + div(npeople, 2), npeople)
        i  = i == 0 ? npeople : i
        i == i_agent && return n_susceptible_contacts
        id = community[i]
        infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
    end
    n_susceptible_contacts
end

function infect_contact!(pr_contact, pr_infect, contact, model, t, n_susceptible_contacts)
    if pr_contact > 0.0 && rand() <= pr_contact         # Contact between agent and contact occurs
        if contact.status == :S && rand() <= pr_infect  # The agent infects the contact
            execute_event!(to_E!, contact, model, t, metrics)
        #elseif contact.status == :R && rand() <= contact.p_reinfection
        #    execute_event!(to_E!, contact, model, t, metrics)
        end
    end
    if contact.status == :S
        n_susceptible_contacts += 1
    end
    n_susceptible_contacts
end

function infect_contactlist!(contactlist::Vector{Int}, pr_contact, pr_infect, agents, model, t, n_susceptible_contacts)
    for id in contactlist
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
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