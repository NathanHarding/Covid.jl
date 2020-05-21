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
using Dates
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

function update_policies!(cfg::Config, dt::Date)
    haskey(cfg.t2distancingregime, dt) && update_struct!(active_distancing_regime, cfg.t2distancingregime[dt])
    haskey(cfg.t2testingregime, dt)    && update_struct!(active_testing_regime,    cfg.t2testingregime[dt])
    haskey(cfg.t2tracingregime, dt)    && update_struct!(active_tracing_regime,    cfg.t2tracingregime[dt])
    haskey(cfg.t2quarantineregime, dt) && update_struct!(active_quarantine_regime, cfg.t2quarantineregime[dt])
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

dummydate() = Date(1900, 1, 1)

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
    dt_last_transition::Date  # Date of most recent status change
    has_been_to_icu::Bool
    has_been_ventilated::Bool
    last_test_date::Date
    last_test_result::Char  # 'p' = positive, 'n' = negative
    quarantined::Bool

    # Risk factors
    age::Int

    # Contact networks
    i_household::Int  # Person is in households[i_household].
    school::Union{Nothing, Vector{Int}}  # Child care, primary school, secondary school, university. Not empty for teachers and students.
    ij_workplace::Union{Nothing, Tuple{Int, Int}}  # Empty for children and teachers (whose workplace is school). person.id == workplaces[i][j].
    i_community::Int  # Shops, transport, pool, library, etc. communitycontacts[i_community] == person.id
    i_social::Int     # Family and/or friends outside the household. socialcontacts[i_social] == person.id
end

Person(id::Int, status::Symbol, age::Int) = Person(id, status, dummydate(), false, false, dummydate(), 'n', false, age, 0, nothing, nothing, 0, 0)

function init_model(indata::Dict{String, DataFrame}, params::T, cfg) where {T <: NamedTuple}
    # Init model
    agedist  = indata["age_distribution"]
    npeople  = round(Int, sum(agedist.count))
    agents   = Vector{Person}(undef, npeople)
    schedule = init_schedule(cfg.firstday, cfg.lastday)
    model    = Model(agents, params, cfg.firstday, cfg.lastday, schedule)

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

function reset_model!(model, cfg)
    firstday       = cfg.firstday
    model.schedule = init_schedule(cfg.firstday, cfg.lastday)  # Empty the schedule
    model.date     = firstday
    agents = model.agents
    params = model.params
    for agent in agents  # Reset each agent's state and schedule a state change
        agent.dt_last_transition = dummydate()
        agent.last_test_date     = dummydate()
        agent.last_test_result  = 'n'
        agent.quarantined       = false
        status = status0[agent.id]
        if status == :S
            agent.status = :S
        elseif status == :E
            to_E!(agent, model, firstday)
        elseif status == :IA
            to_IA!(agent, model, firstday)
        elseif status == :IS
            to_IS!(agent, model, firstday)
        elseif status == :W
            to_W!(agent, model, firstday)
        elseif status == :ICU
            to_ICU!(agent, model, firstday)
        elseif status == :V
            to_V!(agent, model, firstday)
        elseif status == :R
            to_R!(agent, model, firstday)
        elseif status == :D
            to_D!(agent, model, firstday)
        end
    end
end

################################################################################
# Test-trace-quarantine events

function test_for_covid!(agent::Person, model, dt)
    agent.status == :D            && return  # Do not test deceased people
    agent.last_test_result == 'p' && return  # Patient is already a known and active case
    agent.last_test_date == dt    && return  # Patient has already been tested at this time step
    agent.last_test_date = dt
    agent.quarantined = rand() <= active_quarantine_regime.awaiting_test_result.compliance
    schedule!(agent.id, dt + Day(2), get_test_result!, model)  # Test result available 2 days after test
end

function get_test_result!(agent::Person, model, dt)
    agent.quarantined = false
    if (agent.status == :S || agent.status == :R)
        metrics[:negatives] += 1
        agent.last_test_result = 'n'
    else
        metrics[:positives] += 1
        agent.last_test_result = 'p'
        apply_quarantine_regime!(agent, model, dt)
        apply_tracing_regime!(agent, model, dt)
    end
end

function apply_tracing_regime!(agent::Person, model, dt)
    regime = active_tracing_regime
    params = model.params
    contactlist = get_contactlist(agent, :household, params)
    trace_and_test_contacts!(contactlist, model, dt, 1, regime.household.asymptomatic, regime.household.symptomatic)
    contactlist = get_contactlist(agent, :school, params)
    trace_and_test_contacts!(contactlist, model, dt, 2, regime.school.asymptomatic,    regime.school.symptomatic)
    contactlist = get_contactlist(agent, :workplace, params)
    trace_and_test_contacts!(contactlist, model, dt, 3, regime.workplace.asymptomatic, regime.workplace.symptomatic)
    contactlist = get_contactlist(agent, :community, params)
    trace_and_test_contacts!(contactlist, model, dt, 3, regime.community.asymptomatic, regime.community.symptomatic)
    contactlist = get_contactlist(agent, :social, params)
    trace_and_test_contacts!(contactlist, model, dt, 3, regime.social.asymptomatic, regime.social.symptomatic)
end

"Delay is the delay between t and the contact's test date"
function trace_and_test_contacts!(contactlist, model, dt, delay, p_asymptomatic, p_symptomatic)
    p_asymptomatic == 0.0 && p_symptomatic == 0.0 && return
    agents = model.agents
    for contactid in contactlist
        contact = agents[contactid]
        status  = contact.status
        if status == :IS  # Symptomatic and not hospitalised
            rand() <= p_symptomatic && schedule!(contact.id, dt + Day(delay), test_for_covid!, model)
        elseif (status == :S || status == :E || status == :IA || status == :R)  # Asymptomatic
            rand() <= p_asymptomatic && schedule!(contact.id, dt + Day(delay), test_for_covid!, model)
        end
    end
end

"Apply active_quarantine_regime to the agent (who just tested positive) and his/her contacts."
function apply_quarantine_regime!(agent::Person, model, dt)
    # Quarantine the newly positive agent
    regime = active_quarantine_regime
    if rand() <= regime.tested_positive.compliance
        agent.quarantined = true
        status = agent.status
        if status == :IS  # Symptomatic
            dt_exit_quarantine = agent.dt_last_transition + Day(regime.tested_positive.days)  # Days post onset of symptoms
            if dt_exit_quarantine <= dt
                agent.quarantined = false
            else
                schedule!(agent.id, dt_exit_quarantine, exit_quarantine!, model)
            end
        elseif status == :S || status == :E || status == :IA  # Asymptomatic
            schedule!(agent.id, dt + Day(regime.tested_positive.days - 2), exit_quarantine!, model)  # Days post test date
        end
    end

    # Quarantine the newly positive agent's contacts 
    agents = model.agents
    params = model.params
    for (contact_network, quarantine_condition) in regime.case_contacts
        p = quarantine_condition.compliance
        p == 0.0 && continue
        dur = quarantine_condition.days
        dur == 0 && continue
        contactlist = get_contactlist(agent, contact_network, params)
        for contactid in contactlist
            contact = agents[contactid]
            contact.quarantined && continue  # Contact is already quarantined
            status = contact.status
            if (status == :S || status == :E || status == :IA || status == :IS) && rand() <= p
                contact.quarantined = true
                schedule!(agent.id, dt + Day(dur), exit_quarantine!, model)
            end
        end
    end
end

exit_quarantine!(agent::Person, model, dt) = agent.quarantined = false

################################################################################
# State transition events

function to_E!(agent::Person, model, dt)
    agent.status = :E
    agent.dt_last_transition = dt
    dur = max(2, round(Int, dur_incubation(model.params) - 2.0))  # At least 2 days
    schedule!(agent.id, dt + Day(dur), to_IA!, model)
end

function to_IA!(agent::Person, model, dt)
    agent.status = :IA
    agent.dt_last_transition = dt
    params = model.params
    age    = agent.age
    if rand() <= p_IS(params, age)  # Person will progress from IA to IS
        schedule!(agent.id, dt + Day(2), to_IS!, model)
    else  # Person will progress from IA to Recovered
        dur = 2 + dur_IS(params, age)  # Assume asymptomatic duration is the same as pre-hospitalisation symptomatic duration
        schedule!(agent.id, dt + Day(dur), to_R!, model)
    end
    infect_contacts!(agent, model, dt)  # Schedules an immediate status change for each contact
end

function to_IS!(agent::Person, model, dt)
    agent.status = :IS
    agent.dt_last_transition = dt
    params = model.params
    age    = agent.age
    dur    = dur_IS(params, age)
    if rand() < active_testing_regime.IS  # Test for Covid
        dur_totest = max(2, round(Int, dur_onset2test(params)) - 2)  # Time between onset of symptoms and test...at least 2 days
        dur_totest < dur && schedule!(agent.id, dt + Day(dur_totest), test_for_covid!, model)
    end
    if rand() <= p_W(params, age)  # Person will progress from IS to W
        schedule!(agent.id, dt + Day(dur), to_W!, model)
    else  # Person will progress from IS to Recovered
        schedule!(agent.id, dt + Day(dur), to_R!, model)
    end
end

function to_W!(agent::Person, model, dt)
    agent.status = :W
    agent.dt_last_transition = dt
    rand() < active_testing_regime.W && schedule!(agent.id, dt, test_for_covid!, model)  # Test immediately
    params = model.params
    age    = agent.age
    dur    = dur_W(params, age)
    if agent.has_been_to_icu  # Person is improving after ICU - s/he will progress from the ward to recovery after 3 days
        schedule!(agent.id, dt + Day(3), to_R!, model)
    elseif rand() <= p_ICU(params, age)  # Person is deteriorating - s/he will progress from the ward to ICU
        schedule!(agent.id, dt + Day(dur), to_ICU!, model)
    else  # Person will progress from the ward to Recovered without going to ICU
        schedule!(agent.id, dt + Day(dur), to_R!, model)
    end
end

function to_ICU!(agent::Person, model, dt)
    agent.status = :ICU
    agent.dt_last_transition = dt
    agent.has_been_to_icu = true
    params = model.params
    age    = agent.age
    dur    = dur_ICU(params, age)
    if agent.has_been_ventilated  # Person is improving after ventilation - s/he will progress from ICU to the ward after 3 days
        schedule!(agent.id, dt + Day(3), to_W!, model)
    elseif rand() <= p_V(params, age)  # Person is deteriorating - s/he will progress from a non-ventilated ICU bed to a ventilated ICU bed
        schedule!(agent.id, dt + Day(dur), to_V!, model)
    else  # Person will progress from ICU to the ward without going to a ventilated ICU bed
        schedule!(agent.id, dt + Day(dur), to_W!, model)
    end
end

function to_V!(agent::Person, model, dt)
    agent.status = :V
    agent.dt_last_transition = dt
    agent.has_been_ventilated = true
    params = model.params
    age    = agent.age
    dur    = dur_V(params, age)
    if rand() <= p_D(params, age)  # Person will progress from ventilation to deceased
        schedule!(agent.id, dt + Day(dur), to_D!, model)
    else  # Person will progress from ventilation to a non-ventilated ICU bed
        schedule!(agent.id, dt + Day(dur), to_ICU!, model)
    end
end

function to_R!(agent::Person, model, dt)
    agent.status = :R
    agent.dt_last_transition = dt
    agent.last_test_date    = dummydate()
    agent.last_test_result  = 'n'
end

to_D!(agent::Person, model, dt) = agent.status = :D

################################################################################
# Infect contacts event

function infect_contacts!(agent::Person, model, dt)
    agent.status != :IA && agent.status != :IS && return  # Person is either not infectious or isolated in hospital
    agent.quarantined && return  # A quarantined person cannot infect anyone
    dow       = dayofweek(dt)    # 1 = Monday, ..., 7 = Sunday
    agents    = model.agents
    params    = model.params
    pr_infect = p_infect(params, agent.age)
    n_susceptible_contacts = 0
    contactlist = get_contactlist(agent, :household, params)
    n_susceptible_contacts = infect_contactlist!(contactlist, active_distancing_regime.household, pr_infect, agents, model, dt, n_susceptible_contacts)
    if dow <= 5
        contactlist = get_contactlist(agent, :school, params)
        n_susceptible_contacts = infect_contactlist!(contactlist, active_distancing_regime.school,    pr_infect, agents, model, dt, n_susceptible_contacts)
        contactlist = get_contactlist(agent, :workplace, params)
        n_susceptible_contacts = infect_contactlist!(contactlist, active_distancing_regime.workplace, pr_infect, agents, model, dt, n_susceptible_contacts)
    end
    contactlist = get_contactlist(agent, :community, params)
    n_susceptible_contacts = infect_contactlist!(contactlist, active_distancing_regime.community, pr_infect, agents, model, dt, n_susceptible_contacts)
    contactlist = get_contactlist(agent, :social, params)
    n_susceptible_contacts = infect_contactlist!(contactlist, active_distancing_regime.social,    pr_infect, agents, model, dt, n_susceptible_contacts)
    n_susceptible_contacts > 0 && schedule!(agent.id, dt + Day(1), infect_contacts!, model)
end

function infect_contactlist!(contactlist, pr_contact, pr_infect, agents, model, dt, n_susceptible_contacts)
    for id in contactlist
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, agents[id], model, dt, n_susceptible_contacts)
    end
    n_susceptible_contacts
end

function infect_contact!(pr_contact, pr_infect, contact, model, dt, n_susceptible_contacts)
    if !(contact.quarantined) && pr_contact > 0.0 && rand() <= pr_contact         # Contact between agent and contact occurs
        if contact.status == :S && rand() <= pr_infect  # The agent infects the contact
            execute_event!(to_E!, contact, model, dt, metrics)
        #elseif contact.status == :R && rand() <= contact.p_reinfection
        #    execute_event!(to_E!, contact, model, dt, metrics)
        end
    end
    if contact.status == :S
        n_susceptible_contacts += 1
    end
    n_susceptible_contacts
end

################################################################################
# Functions dependent on model parameters; called by event functions

# Durations
dur_incubation(params) = rand(LogNormal(params.mu_incubation, params.sigma_incubation))  # Duration of E + IA
dur_onset2test(params) = rand(Gamma(params.shape_onset2test, params.scale_onset2test))
dur_IS(params,  age)   = rand(Poisson(exp(params.b0_IS  + params.b1_IS  * age)))
dur_W(params,   age)   = rand(Poisson(exp(params.b0_W   + params.b1_W   * age)))
dur_ICU(params, age)   = rand(Poisson(exp(params.b0_ICU + params.b1_ICU * age)))
dur_V(params,   age)   = rand(Poisson(exp(params.b0_V   + params.b1_V   * age)))

# Transition probabilities

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

p_IS(params, age)  = 1.0 / (1.0 + exp(-(params.a0_IS  + params.a1_IS  * age)))  # Pr(Next state is IS  | Current state is IA,  age)
p_ICU(params, age) = 1.0 / (1.0 + exp(-(params.a0_ICU + params.a1_ICU * age)))  # Pr(Next state is ICU | Current state is W,   age)
p_V(params, age)   = 1.0 / (1.0 + exp(-(params.a0_V   + params.a1_V   * age)))  # Pr(Next state is V   | Current state is ICU, age)
p_D(params, age)   = 1.0 / (1.0 + exp(-(params.a0_D   + params.a1_D   * age)))  # Pr(Next state is D   | Current state is V,   age)

p_infect(params, age) = 1.0 / (1.0 + exp(-(params.a0_infect + params.a1_infect * age)))  # Pr(Infect contact | age)

################################################################################
# Utils

function get_contactlist(agent::Person, network::Symbol, params)
    agentid   = agent.id
    ncontacts = 0
    if network == :household
        ncontacts = get_household_contactids!(households[agent.i_household], agentid)
    elseif network == :school
        ncontacts = isnothing(agent.school) ? 0 : get_school_contactids!(agent.school)
    elseif network == :workplace
        if !isnothing(agent.ij_workplace)
            i, j = agent.ij_workplace
            ncontacts = get_community_contactids!(workplaces[i], j, Int(params.n_workplace_contacts), agentid)
        end
    elseif network == :community
        ncontacts = get_community_contactids!(communitycontacts, agent.i_community, Int(params.n_community_contacts), agentid)
    elseif network == :social
        ncontacts = get_community_contactids!(socialcontacts, agent.i_social, Int(params.n_social_contacts), agentid)
    end
    view(contactids, 1:ncontacts)
end

function get_household_contactids!(household, agentid)
    j = 0
    flds = (:adults, :children)
    for fld in flds
        contactlist = getfield(household, fld)
        for id in contactlist
            id == agentid && continue
            j += 1
            contactids[j] = id
        end
    end
    j
end

function get_school_contactids!(contactlist::Vector{Int})
    ncontacts = length(contactlist)
    for j = 1:ncontacts
        contactids[j] = contactlist[j]
    end
    ncontacts
end

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