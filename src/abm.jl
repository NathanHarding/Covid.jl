"""
SEIHCVRD age-specific transition probabilities and durations.
The age structure is read from disk.

Pr(next state is j | current state is i, age) = 1 / (1 + exp(-(a0 + a1*age)))

duration|age ~ Poisson(lambda(age)), where lambda(age) = exp(b0 + b1*age)
"""
module abm

export Config, metrics, update_policies!, init_model, reset_model!, reset_metrics!,  # API required from any model module
       init_output, reset_output!, execute_events!, metrics_to_output!  # Re-exported from the core module as is

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

################################################################################
# Policies

const active_distancing_regime = DistancingRegime(0.0, 0.0, 0.0, 0.0, 0.0)
const active_testing_regime    = TestingRegime(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
const active_tracing_regime    = TracingRegime((asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0))
const active_quarantine_regime = QuarantineRegime((days=0,compliance=0.0), (days=0,compliance=0.0), Dict(:household => (days=0,compliance=0.0)))

function update_policies!(cfg::Config, t::Int)
    haskey(cfg.t2distancingregime, t) && update_struct!(active_distancing_regime, cfg.t2distancingregime[t])
    haskey(cfg.t2testingregime, t)    && update_struct!(active_testing_regime,    cfg.t2testingregime[t])
    haskey(cfg.t2tracingregime, t)    && update_struct!(active_tracing_regime,    cfg.t2tracingregime[t])
    haskey(cfg.t2quarantineregime, t) && update_struct!(active_quarantine_regime, cfg.t2quarantineregime[t])
end

################################################################################
# Metrics

const metrics = Dict(:S => 0, :E => 0, :IA => 0, :IS => 0, :W => 0, :ICU => 0, :V => 0, :R => 0, :D => 0,
                     :positives => 0, :negatives => 0)

function reset_metrics!(model)
    for (k, v) in metrics
        metrics[k] = 0
    end
    for agent in model.agents
        metrics[agent.status] += 1
    end
end

"Remove the agent's old state from metrics"
function unfit!(metrics::Dict{Symbol, Int}, agent)
    metrics[agent.status] -= 1
end

"Add the agent's new state to metrics"
function fit!(metrics::Dict{Symbol, Int}, agent)
    metrics[agent.status] += 1
end

################################################################################
# Conveniences
const status0    = Symbol[]       # Used when resetting the model at the beginning of a run
const contactids = fill(0, 1000)  # Buffer for a mutable contact list
const households = Household[]    # households[i].adults[j] is the id of the jth adult in the ith household. Ditto children.
const workplaces = Vector{Int}[]  # workplaces[i][j] is the id of the jth worker in the ith workplace.
const communitycontacts = Int[]   # Contains person IDs. Community contacts can be derived for each person.
const socialcontacts    = Int[]   # Contains person IDs. Social contacts can be derived for each person.

################################################################################
# The model

mutable struct Person <: AbstractAgent
    id::Int
    status::Symbol  # S, E, I1, I2, I3, H, C, V, R, D
    t_last_transition::Int  # Time of most recent status change
    has_been_to_icu::Bool
    has_been_ventilated::Bool
    last_test_time::Int
    last_test_result::Char  # 'p' = positive, 'n' = negative
    quarantined::Bool

    # Risk factors
    age::Int

    # Contacts
    i_household::Int  # Person is in households[i_household].
    school::Union{Nothing, Vector{Int}}  # Child care, primary school, secondary school, university. Not empty for teachers and students.
    ij_workplace::Union{Nothing, Tuple{Int, Int}}  # Empty for children and teachers (whose workplace is school). person.id == workplaces[i][j].
    i_community::Int  # Shops, transport, pool, library, etc. communitycontacts[i_community] == person.id
    i_social::Int     # Family and/or friends outside the household. socialcontacts[i_social] == person.id
end

Person(id::Int, status::Symbol, age::Int) = Person(id, status, -1, false, false, -1, 'n', false, age, 0, nothing, nothing, 0, 0)

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
            agents[id].status != :S && continue  # We've already reset this person's status from the default status S
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
        agent.t_last_transition = -1
        agent.last_test_time    = -1
        agent.last_test_result  = 'n'
        agent.quarantined       = false
        status = status0[agent.id]
        if status == :S
            agent.status = :S
        elseif status == :E
            to_E!(agent, model, 0)
        elseif status == :IA
            to_IA!(agent, model, 0)
        elseif status == :IS
            to_IS!(agent, model, 0)
        elseif status == :W
            to_W!(agent, model, 0)
        elseif status == :ICU
            to_ICU!(agent, model, 0)
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
# Events

function test_for_covid!(agent::Person, model, t)
    agent.status == :D            && return  # Do not test deceased people
    agent.last_test_result == 'p' && return  # Patient is already a known and active case
    agent.last_test_time == t     && return  # Patient has already been tested at this time step
    agent.last_test_time = t
    agent.quarantined = rand() <= active_quarantine_regime.awaiting_test_result.compliance
    schedule!(agent.id, t + 2, get_test_result!, model)  # Test result available 2 days after test
end

function get_test_result!(agent::Person, model, t)
    agent.quarantined = false
    if (agent.status == :S || agent.status == :R)
        metrics[:negatives] += 1
        agent.last_test_result = 'n'
    else
        metrics[:positives] += 1
        agent.last_test_result = 'p'
        apply_quarantine_regime!(agent, model, t)
        trace_and_test_contacts!(agent, model, t)
    end
end

function trace_and_test_contacts!(agent::Person, model, t)
    regime  = active_tracing_regime
    params  = model.params
    agentid = agent.id
    trace_household_contacts!(agent, model, t, regime.household, 1)  # Last arg is the delay between t and the contact's test date
    trace_school_contacts!(agent::Person, model, t, regime.school, 2)
    if !isnothing(agent.ij_workplace)
        i, j = agent.ij_workplace
        trace_community_contacts!(agentid, model, t, regime.workplace.asymptomatic, regime.workplace.symptomatic, 3,
                                  workplaces[i], j, Int(params.n_workplace_contacts))
    end
    trace_community_contacts!(agentid, model, t, regime.community.asymptomatic, regime.community.symptomatic, 3,
                              communitycontacts, agent.i_community, Int(params.n_community_contacts))
    trace_community_contacts!(agentid, model, t, regime.social.asymptomatic, regime.social.symptomatic, 3,
                              socialcontacts, agent.i_social, Int(params.n_social_contacts))
end

function trace_and_test_contact!(contact::Person, model, t, delay, p_asymptomatic, p_symptomatic)
    status = contact.status
    if status == :IS  # Symptomatic and not hospitalised
        rand() <= p_symptomatic && schedule!(contact.id, t + delay, test_for_covid!, model)
    elseif (status == :S || status == :E || status == :IA || status == :R)  # Asymptomatic
        rand() <= p_asymptomatic && schedule!(contact.id, t + delay, test_for_covid!, model)
    end
end

function trace_household_contacts!(agent::Person, model, t, probs, delay)
    p_asymptomatic = probs.asymptomatic
    p_symptomatic  = probs.symptomatic
    p_asymptomatic == 0.0 && p_symptomatic == 0.0 && return
    agents = model.agents
    household = households[agent.i_household]
    flds = (:adults, :children)
    for fld in flds
        contactids = getfield(household, fld)
        for contactid in contactids
            trace_and_test_contact!(agents[contactid], model, t, delay, p_asymptomatic, p_symptomatic)
        end
    end
end

function trace_school_contacts!(agent::Person, model, t, probs, delay)
    isnothing(agent.school) && return  # Agent has no school contacts
    p_asymptomatic = probs.asymptomatic
    p_symptomatic  = probs.symptomatic
    p_asymptomatic == 0.0 && p_symptomatic == 0.0 && return
    agents = model.agents
    contactids = agent.school
    for contactid in contactids
        trace_and_test_contact!(agents[contactid], model, t, delay, p_asymptomatic, p_symptomatic)
    end
end

function trace_community_contacts!(agentid::Int, model, t, p_asymptomatic, p_symptomatic, delay, community::Vector{Int}, i_agent::Int, ncontacts_per_person::Int)
    i_agent == 0 && return
    p_asymptomatic == 0.0 && p_symptomatic == 0.0 && return
    ncontacts = get_community_contactids!(community, i_agent, ncontacts_per_person, agentid)
    ncontacts == 0 && return
    for j = 1:ncontacts
        trace_and_test_contact!(agents[contactids[j]], model, t, delay, p_asymptomatic, p_symptomatic)
    end
end

"Apply active_quarantine_regime to the agent (who just tested positive) and his/her contacts."
function apply_quarantine_regime!(agent::Person, model, t)
    # Quarantine the newly positive agent
    regime = active_quarantine_regime
    if rand() <= regime.tested_positive.compliance
        agent.quarantined = true
        status = agent.status
        if status == :IS  # Symptomatic
            t_exit_quarantine = agent.t_last_transition + regime.tested_positive.days  # Days post onset of symptoms
            if t_exit_quarantine <= t
                agent.quarantined = false
            else
                schedule!(agent.id, t_exit_quarantine, exit_quarantine!, model)
            end
        elseif status == :S || status == :E || status == :IA  # Asymptomatic
            schedule!(agent.id, t + regime.tested_positive.days - 2, exit_quarantine!, model)  # Days post test date
        end
    end

    # Quarantine the newly positive agent's contacts 
    agents = model.agents
    for (contact_network, quarantine_condition) in regime.case_contacts
        p = quarantine_condition.compliance
        p == 0.0 && continue
        dur = quarantine_condition.days
        dur == 0 && continue
        contactids = getfield(agent, contact_network)
        for contactid in contactids
            contact = agents[contactid]
            contact.quarantined && continue  # Contact is already quarantined
            status = contact.status
            if (status == :S || status == :E || status == :IA || status == :IS) && rand() <= p
                contact.quarantined = true
                schedule!(agent.id, t + dur, exit_quarantine!, model)
            end
        end
    end
end

exit_quarantine!(agent::Person, model, t) = agent.quarantined = false

function to_E!(agent::Person, model, t)
    agent.status = :E
    agent.t_last_transition = t
    dur = dur_E(model.params, agent.age)
    schedule!(agent.id, t + dur, to_IA!, model)
end

function to_IA!(agent::Person, model, t)
    agent.status = :IA
    agent.t_last_transition = t
    params = model.params
    age    = agent.age
    if rand() <= p_IS(params, age)  # Person will progress from IA to IS
        dur = dur_IA(params, age)
        schedule!(agent.id, t + dur, to_IS!, model)
    else  # Person will progress from IA to Recovered
        dur = dur_IA(params, age) + dur_IS(params, age)  # Assume asymptomatic duration is the same as pre-hospitalisation duration
        schedule!(agent.id, t + dur, to_R!, model)
    end
    infect_contacts!(agent, model, t)  # Schedules an immediate status change for each contact
end

function to_IS!(agent::Person, model, t)
    agent.status = :IS
    agent.t_last_transition = t
    rand() < active_testing_regime.IS && schedule!(agent.id, t + 2, test_for_covid!, model)  # Test 2 days after onset of symptoms
    params = model.params
    age = agent.age
    dur = dur_IS(params, age)
    if rand() <= p_W(params, age)  # Person will progress from IS to W
        schedule!(agent.id, t + dur, to_W!, model)
    else  # Person will progress from IS to Recovered
        schedule!(agent.id, t + dur, to_R!, model)
    end
end

function to_W!(agent::Person, model, t)
    agent.status = :W
    agent.t_last_transition = t
    rand() < active_testing_regime.W && schedule!(agent.id, t, test_for_covid!, model)  # Test immediately
    params = model.params
    age    = agent.age
    dur    = dur_W(params, age)
    if agent.has_been_to_icu  # Person is improving after ICU - s/he will progress from the ward to recovery after 3 days
        schedule!(agent.id, t + 3, to_R!, model)
    elseif rand() <= p_ICU(params, age)  # Person is deteriorating - s/he will progress from the ward to ICU
        schedule!(agent.id, t + dur, to_ICU!, model)
    else  # Person will progress from the ward to Recovered without going to ICU
        schedule!(agent.id, t + dur, to_R!, model)
    end
end

function to_ICU!(agent::Person, model, t)
    agent.status = :ICU
    agent.t_last_transition = t
    agent.has_been_to_icu = true
    params = model.params
    age    = agent.age
    dur    = dur_ICU(params, age)
    if agent.has_been_ventilated  # Person is improving after ventilation - s/he will progress from ICU to the ward after 3 days
        schedule!(agent.id, t + 3, to_W!, model)
    elseif rand() <= p_V(params, age)  # Person is deteriorating - s/he will progress from a non-ventilated ICU bed to a ventilated ICU bed
        schedule!(agent.id, t + dur, to_V!, model)
    else  # Person will progress from ICU to the ward without going to a ventilated ICU bed
        schedule!(agent.id, t + dur, to_W!, model)
    end
end

function to_V!(agent::Person, model, t)
    agent.status = :V
    agent.t_last_transition = t
    agent.has_been_ventilated = true
    params = model.params
    age    = agent.age
    dur    = dur_V(params, age)
    if rand() <= p_D(params, age)  # Person will progress from ventilation to deceased
        schedule!(agent.id, t + dur, to_D!, model)
    else  # Person will progress from ventilation to a non-ventilated ICU bed
        schedule!(agent.id, t + dur, to_ICU!, model)
    end
end

function to_R!(agent::Person, model, t)
    agent.status = :R
    agent.t_last_transition = t
    agent.last_test_time    = -1
    agent.last_test_result  = 'n'
end

to_D!(agent::Person, model, t) = agent.status = :D

################################################################################
# Functions dependent on model parameters; called by event functions

# Duration of each state
dur_E(params,   age) = rand(Poisson(exp(params.b0_E   + params.b1_E   * age)))
dur_IA(params,  age) = rand(Poisson(exp(params.b0_IA  + params.b1_IA  * age)))
dur_IS(params,  age) = rand(Poisson(exp(params.b0_IS  + params.b1_IS  * age)))
dur_W(params,   age) = rand(Poisson(exp(params.b0_W   + params.b1_W   * age)))
dur_ICU(params, age) = rand(Poisson(exp(params.b0_ICU + params.b1_ICU * age)))
dur_V(params,   age) = rand(Poisson(exp(params.b0_V   + params.b1_V   * age)))

"Pr(Next state is W | Current state is IS, age)."
function p_W(params, age)
    age <= 9  && return params.p_W_0to9
    age <= 19 && return params.p_W_10to19
    age <= 29 && return params.p_W_20to29
    age <= 39 && return params.p_W_30to39
    age <= 49 && return params.p_W_40to49
    age <= 59 && return params.p_W_50to59
    age <= 69 && return params.p_W_60to69
    age <= 79 && return params.p_W_70to79
    params.p_W_gte80
end

# Transition probabilities
p_IS(params, age)  = 1.0 / (1.0 + exp(-(params.a0_IS  + params.a1_IS  * age)))  # Pr(Next state is IS  | Current state is IA,  age)
p_ICU(params, age) = 1.0 / (1.0 + exp(-(params.a0_ICU + params.a1_ICU * age)))  # Pr(Next state is ICU | Current state is W,   age)
p_V(params, age)   = 1.0 / (1.0 + exp(-(params.a0_V   + params.a1_V   * age)))  # Pr(Next state is V   | Current state is ICU, age)
p_D(params, age)   = 1.0 / (1.0 + exp(-(params.a0_D   + params.a1_D   * age)))  # Pr(Next state is D   | Current state is V,   age)

p_infect(params, age) = 1.0 / (1.0 + exp(-(params.a0_infect + params.a1_infect * age)))  # Pr(Infect contact | age)

################################################################################
# Infect contacts

function infect_contacts!(agent::Person, model, t::Int)
    agent.status != :IA && agent.status != :IS && return
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
        n_susceptible_contacts = infect_community_contacts!(workplaces[i], j, Int(params.n_workplace_contacts), agentid,
                                                            active_distancing_regime.workplace, pr_infect, agents, model, t, n_susceptible_contacts)
    end
    n_susceptible_contacts = infect_community_contacts!(communitycontacts, agent.i_community, Int(params.n_community_contacts), agentid,
                                                        active_distancing_regime.community, pr_infect, agents, model, t, n_susceptible_contacts)
    n_susceptible_contacts = infect_community_contacts!(socialcontacts, agent.i_social, Int(params.n_social_contacts), agentid,
                                                        active_distancing_regime.social, pr_infect, agents, model, t, n_susceptible_contacts)
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

function infect_community_contacts!(community::Vector{Int}, i_agent::Int, ncontacts_per_person::Int, agentid::Int,
                                    pr_contact, pr_infect, agents, model, t, n_susceptible_contacts)
    ncontacts = get_community_contactids!(community, i_agent, ncontacts_per_person, agentid)
    ncontacts == 0 && return n_susceptible_contacts
    infect_contactlist!(view(contactids, 1:ncontacts), pr_contact, pr_infect, agents, model, t, n_susceptible_contacts)
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

function infect_contactlist!(contactlist, pr_contact, pr_infect, agents, model, t, n_susceptible_contacts)
    for id in contactlist
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, t, n_susceptible_contacts)
    end
    n_susceptible_contacts
end

################################################################################
# Utils

"""
Modified: contactids.

Populate contactids (global) with the agent's contact IDs and return the number of contacts j.
I.e., contactids[1:j] is the required contact list.
"""
function get_community_contactids!(community::Vector{Int}, i_agent::Int, ncontacts_per_person::Int, agentid::Int)
    i_agent == 0 && return 0
    j = 0  # Index of contactids
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
            j += 1
            contactids[j] = community[i]
        end
    elseif i1 > i2
        for i = i1:npeople
            i == i_agent && continue
            j += 1
            contactids[j] = community[i]
        end
        for i = 1:i2
            i == i_agent && continue
            j += 1
            contactids[j] = community[i]
        end
    end
    if isodd(ncontacts_per_person)
        i  = rem(i_agent + div(npeople, 2), npeople)
        i  = i == 0 ? npeople : i
        i == i_agent && return j
        j += 1
        contactids[j] = community[i]
    end
    j
end

end