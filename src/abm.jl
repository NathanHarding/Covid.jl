"""
SEIHCVRD age-specific transition probabilities and durations.
The age structure is read from disk.

Pr(next state is j | current state is i, age) = 1 / (1 + exp(-(a0 + a1*age)))

duration|age ~ Poisson(lambda(age)), where lambda(age) = exp(b0 + b1*age)
"""
module abm

export Config, metrics, update_policies!, init_model, reset_model!, reset_metrics!, apply_forcing!, # API required from any model module
       init_output, reset_output!, execute_events!, metrics_to_output!  # Re-exported from the core module as is

using DataFrames
using Dates
using Demographics
using Distributions

include("core.jl")
using .core
import .core: unfit!  # To be extended by model-specific method
import .core: fit!    # To be extended by model-specific method

# Model-specific dependencies
include("config.jl")
using .config

################################################################################
# Agent state

mutable struct DiseaseProgression
    status::Symbol  # S, E, IA, IS, H, W, ICU, V, R, D
    age::Int
    dt_last_transition::Date  # Date of most recent status change
    most_severe_state::Symbol
    infectious_start::Date
    infectious_end::Date
    incubation_end::Date
    symptoms_end::Date
    last_test_date::Date
    last_test_result::Char  # 'p' = positive, 'n' = negative
    quarantined::Bool
end

DiseaseProgression(age) = DiseaseProgression(:S, age, dummydate(), :X, dummydate(), dummydate(), dummydate(), dummydate(), dummydate(), 'n', false)

function reset_state!(state::DiseaseProgression)
    date0 = dummydate()
    state.status             = :S
    state.dt_last_transition = date0
    state.most_severe_state  = :X
    state.infectious_start   = date0
    state.infectious_end     = date0
    state.incubation_end     = date0
    state.symptoms_end       = date0
    state.last_test_date     = date0
    state.last_test_result   = 'n'
    state.quarantined        = false
end

################################################################################
# Policies

const active_distancing_policy = DistancingPolicy(0.0, 0.0, 0.0, 0.0, 0.0)
const active_testing_policy    = TestingPolicy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
const active_tracing_policy    = TracingPolicy((asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0), (asymptomatic=0.0,symptomatic=0.0))
const active_quarantine_policy = QuarantinePolicy((days=0,compliance=0.0), (days=0,compliance=0.0), Dict(:household => (days=0,compliance=0.0)))

function update_policies!(cfg::Config, dt::Date)
    haskey(cfg.t2distancingpolicy, dt) && update_struct!(active_distancing_policy, cfg.t2distancingpolicy[dt])
    haskey(cfg.t2testingpolicy, dt)    && update_struct!(active_testing_policy,    cfg.t2testingpolicy[dt])
    haskey(cfg.t2tracingpolicy, dt)    && update_struct!(active_tracing_policy,    cfg.t2tracingpolicy[dt])
    haskey(cfg.t2quarantinepolicy, dt) && update_struct!(active_quarantine_policy, cfg.t2quarantinepolicy[dt])
end

################################################################################
# Metrics

const metrics = Dict(:S => 0, :E => 0, :IA => 0, :IS => 0, :H => 0, :W => 0, :ICU => 0, :V => 0, :R => 0, :D => 0,
                     :positives => 0, :negatives => 0)

function reset_metrics!(model)
    for (k, v) in metrics
        metrics[k] = 0
    end
    for agent in model.agents
        metrics[agent.state.status] += 1
    end
end

"Remove the agent's old state from metrics"
function unfit!(metrics::Dict{Symbol, Int}, agent)
    metrics[agent.state.status] -= 1
end

"Add the agent's new state to metrics"
function fit!(metrics::Dict{Symbol, Int}, agent)
    metrics[agent.state.status] += 1
end

################################################################################
# Conveniences

const most_severe_states = [:IA, :H, :W, :ICU, :V, :D]
const lb2dist = Dict{Int, Categorical}()  # agegroup_lb => Distribution of most severe state. Determines the path the the person is on.

dummydate() = Date(1900, 1, 1)

function update_lb2dist!(params)
    lb2dist[0]  = Categorical([params[:p_IA_0to9],   params[:p_H_0to9],   params[:p_W_0to9],   params[:p_ICU_0to9],   params[:p_V_0to9],   params[:p_death_0to9]])
    lb2dist[10] = Categorical([params[:p_IA_10to19], params[:p_H_10to19], params[:p_W_10to19], params[:p_ICU_10to19], params[:p_V_10to19], params[:p_death_10to19]])
    lb2dist[20] = Categorical([params[:p_IA_20to29], params[:p_H_20to29], params[:p_W_20to29], params[:p_ICU_20to29], params[:p_V_20to29], params[:p_death_20to29]])
    lb2dist[30] = Categorical([params[:p_IA_30to39], params[:p_H_30to39], params[:p_W_30to39], params[:p_ICU_30to39], params[:p_V_30to39], params[:p_death_30to39]])
    lb2dist[40] = Categorical([params[:p_IA_40to49], params[:p_H_40to49], params[:p_W_40to49], params[:p_ICU_40to49], params[:p_V_40to49], params[:p_death_40to49]])
    lb2dist[50] = Categorical([params[:p_IA_50to59], params[:p_H_50to59], params[:p_W_50to59], params[:p_ICU_50to59], params[:p_V_50to59], params[:p_death_50to59]])
    lb2dist[60] = Categorical([params[:p_IA_60to69], params[:p_H_60to69], params[:p_W_60to69], params[:p_ICU_60to69], params[:p_V_60to69], params[:p_death_60to69]])
    lb2dist[70] = Categorical([params[:p_IA_70to79], params[:p_H_70to79], params[:p_W_70to79], params[:p_ICU_70to79], params[:p_V_70to79], params[:p_death_70to79]])
    lb2dist[80] = Categorical([params[:p_IA_gte80],  params[:p_H_gte80],  params[:p_W_gte80],  params[:p_ICU_gte80],  params[:p_V_gte80],  params[:p_death_gte80]])
end

"Forcibly change a person's status from Susceptible according to cfg.forcing[dt]"
function apply_forcing!(forcing, model, dt)
    !haskey(forcing, dt) && return nothing
    status2n = forcing[dt]
    agents   = model.agents
    npeople  = length(agents)
    rg = 1:npeople
    for (status, n) in status2n
        nsuccesses = 0
        jmax = 10 * npeople  # Ceiling on the number of iterations
        for j = 1:jmax
            id = rand(rg)
            agent = agents[id]
            agent.state.status != :S && continue  # This person's status is already not Susceptible
            if status == :E
                execute_event!(to_E!, agent, model, dt, metrics)
            elseif status == :IA
                execute_event!(to_IA!, agent, model, dt, metrics)
            elseif status == :IS
                execute_event!(to_IS!, agent, model, dt, metrics)
            elseif status == :H
                execute_event!(to_H!, agent, model, dt, metrics)
            elseif status == :W
                execute_event!(to_W!, agent, model, dt, metrics)
            elseif status == :ICU
                execute_event!(to_ICU!, agent, model, dt, metrics)
            elseif status == :V
                execute_event!(to_V!, agent, model, dt, metrics)
            elseif status == :R
                execute_event!(to_R!, agent, model, dt, metrics)
            elseif status == :D
                execute_event!(to_D!, agent, model, dt, metrics)
            end
            nsuccesses += 1
            nsuccesses == n && break
        end
    end
    nothing
end

################################################################################
# The model

function init_model(params::Dict{Symbol, Float64}, cfg)
    # Set conveniences
    update_lb2dist!(params)

    # Construct people with state
    people  = construct_population(cfg.demographics)
    npeople = size(people, 1)
    agents  = Vector{Person{Char, DiseaseProgression}}(undef, npeople)
    dt      = today()
    for id = 1:npeople
        person     = people[id]
        age        = Demographics.age(person, dt, :year)
        state      = DiseaseProgression(age)
        agents[id] = Person{Char, DiseaseProgression}(person.id, person.birthdate, person.sex, person.address, state,
                                                      person.i_household, person.school, person.ij_workplace, person.i_community, person.i_social)
    end

    # Init model
    schedule = init_schedule(cfg.firstday, cfg.lastday)
    Model(agents, params, cfg.firstday, cfg.lastday, schedule)
end

function reset_model!(model, cfg)
    model.schedule = init_schedule(cfg.firstday, cfg.lastday)  # Empty the schedule
    model.date     = cfg.firstday
    agents = model.agents
    for agent in agents
        reset_state!(agent.state)
    end
end

################################################################################
# Test-trace-quarantine events

function test_for_covid!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status == :D            && return  # Do not test deceased people
    state.last_test_result == 'p' && return  # Patient is already a known and active case
    state.last_test_date == dt    && return  # Patient has already been tested at this time step
    state.last_test_date = dt
    state.quarantined = rand() <= active_quarantine_policy.awaiting_test_result.compliance
    schedule!(agent.id, dt + Day(2), get_test_result!, model)  # Test result available 2 days after test
end

function get_test_result!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.quarantined = false
    if (state.status == :S || state.status == :R)
        metrics[:negatives] += 1
        state.last_test_result = 'n'
    else
        metrics[:positives] += 1
        state.last_test_result = 'p'
        apply_quarantine_policy!(agent, model, dt)
        apply_tracing_policy!(agent, model, dt)
    end
end

function apply_tracing_policy!(agent::Person{Char, DiseaseProgression}, model, dt)
    policy = active_tracing_policy
    params = model.params
    ncontacts = get_contactlist(agent, :household, params)
    trace_and_test_contacts!(ncontacts, model, dt, 1, policy.household.asymptomatic, policy.household.symptomatic)
    ncontacts = get_contactlist(agent, :school, params)
    trace_and_test_contacts!(ncontacts, model, dt, 2, policy.school.asymptomatic,    policy.school.symptomatic)
    ncontacts = get_contactlist(agent, :workplace, params)
    trace_and_test_contacts!(ncontacts, model, dt, 3, policy.workplace.asymptomatic, policy.workplace.symptomatic)
    ncontacts = get_contactlist(agent, :community, params)
    trace_and_test_contacts!(ncontacts, model, dt, 3, policy.community.asymptomatic, policy.community.symptomatic)
    ncontacts = get_contactlist(agent, :social, params)
    trace_and_test_contacts!(ncontacts, model, dt, 3, policy.social.asymptomatic, policy.social.symptomatic)
end

"Delay is the delay between t and the contact's test date"
function trace_and_test_contacts!(ncontacts, model, dt, delay, p_asymptomatic, p_symptomatic)
    p_asymptomatic == 0.0 && p_symptomatic == 0.0 && return
    agents = model.agents
    for i = 1:ncontacts
        contactid = getcontact(i)
        contact   = agents[contactid]
        status    = contact.state.status
        if status == :IS  # Symptomatic and not hospitalised
            rand() <= p_symptomatic && schedule!(contact.id, dt + Day(delay), test_for_covid!, model)
        elseif (status == :S || status == :E || status == :IA || status == :R)  # Asymptomatic
            rand() <= p_asymptomatic && schedule!(contact.id, dt + Day(delay), test_for_covid!, model)
        end
    end
end

"Apply active_quarantine_policy to the agent (who just tested positive) and his/her contacts."
function apply_quarantine_policy!(agent::Person{Char, DiseaseProgression}, model, dt)
    # Quarantine the newly positive agent
    policy = active_quarantine_policy
    if rand() <= policy.tested_positive.compliance
        state = agent.state
        state.quarantined = true
        status = state.status
        if status == :IS || status == :H  # Symptomatic and not hospitalised
            dt_exit_quarantine = state.dt_last_transition + Day(policy.tested_positive.days)  # Days post onset of symptoms
            if dt_exit_quarantine <= dt
                state.quarantined = false
            else
                schedule!(agent.id, dt_exit_quarantine, exit_quarantine!, model)
            end
        elseif status == :S || status == :E || status == :IA  # Asymptomatic
            schedule!(agent.id, dt + Day(policy.tested_positive.days - 2), exit_quarantine!, model)  # Days post test date
        end
    end

    # Quarantine the newly positive agent's contacts 
    agents = model.agents
    params = model.params
    for (contact_network, quarantine_condition) in policy.case_contacts
        p = quarantine_condition.compliance
        p == 0.0 && continue
        dur = quarantine_condition.days
        dur == 0 && continue
        ncontacts = get_contactlist(agent, contact_network, params)
        for i = 1:ncontacts
            contactid = getcontact(i)
            contact   = agents[contactid]
            state     = contact.state
            state.quarantined && continue  # Contact is already quarantined
            status = state.status
            if (status == :S || status == :E || status == :IA || status == :IS || status == :H) && rand() <= p
                state.quarantined = true
                schedule!(agent.id, dt + Day(dur), exit_quarantine!, model)
            end
        end
    end
end

function exit_quarantine!(agent::Person{Char, DiseaseProgression}, model, dt)
    status = agent.state.status
    agent.state.quarantined = (status == :W || status == :ICU || status == :V)  # Remain quarantined if hospitalised, else exit quarantine
end

################################################################################
# State transition events

"Start of the incubation period."
function to_E!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :E
    state.dt_last_transition = dt
    days_incubating          = max(2, dur_incubation(model.params))  # Time from exposure to symptoms (if sympmtoms occur); at least 2 days
    days_to_infectious       = max(1, days_incubating - 2)           # Time from exposure to infectious; at least 1 day; starts 1 or 2 days before incubation period ends
    state.infectious_start   = dt + Day(days_to_infectious)
    state.incubation_end     = dt + Day(days_incubating)
    schedule!(agent.id, state.infectious_start, to_IA!, model)  # Person becomes IA before the incubation period ends
end

"Start of the infectious period, 1-2 days before the end of the incubation period."
function to_IA!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :IA
    state.dt_last_transition = dt
    state.most_severe_state  = draw_most_severe_state(state.age)  # IA, Symp, W, ICU, V, D
    days_to_end_incubation   = (state.incubation_end - dt).value
    days_infectious          = max(days_to_end_incubation + 1, dur_infectious(model.params))  # Ensures patient is still infectious after incubation period
    state.infectious_end     = dt + Day(days_infectious)
    if state.most_severe_state == :IA
        schedule!(agent.id, state.infectious_end, to_R!, model)   # Person remains IA and recovers upon exiting the infectious period
    else
        schedule!(agent.id, state.incubation_end, to_IS!, model)  # Person becomes symptomatic after the incubation period
    end
    infect_contacts!(agent, model, dt)
end

"End of the incubation period. Start symptomatic period."
function to_IS!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :IS
    state.dt_last_transition = dt
    most_severe_state     = state.most_severe_state
    days_to_noninfectious = (state.infectious_end - dt).value
    dur_symptomatic       = draw_total_duration_of_symptoms(state.age, most_severe_state, model.params) # Total duration of symptoms
    dur_symptomatic       = max(days_to_noninfectious, dur_symptomatic)  # Symptoms end on or after the end of the infectious period
    state.symptoms_end    = dt + Day(dur_symptomatic)
    if most_severe_state == :H        # Path is: Home->Recovery
        dur_home = dur_symptomatic - days_to_noninfectious
    elseif most_severe_state == :W    # Path is: Home->Ward->Recovery
        dur_home = round(Int, 0.4 * dur_symptomatic) - days_to_noninfectious   # Split dur_symptomatic: 40% Home, 60% Ward
    elseif most_severe_state == :ICU  # Path is: Home->Ward->ICU->Ward->Recovery
        dur_home = round(Int, 0.25 * dur_symptomatic) - days_to_noninfectious  # Split dur_symptomatic: 25% Home, 40% Ward, 35% ICU
    elseif most_severe_state == :V    # Path is: Home->Ward->ICU->Vent->ICU->Ward->Recovery
        dur_home = round(Int, 0.2 * dur_symptomatic) - days_to_noninfectious   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
    else  # most_severe_state == :D   # Path is: Home->Ward->ICU->Vent->deceased
        dur_home = round(Int, 0.2 * dur_symptomatic) - days_to_noninfectious   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
    end
    dur_home  = max(0, dur_home)
    dt_exit_H = dt + Day(dur_home)
    if most_severe_state == :H  # Person will not be hospitalised
        if dt_exit_H > state.infectious_end  # Symptoms persist after infectious period -> proceed from IS to H
            schedule!(agent.id, state.infectious_end, to_H!, model)
        else
            schedule!(agent.id, state.symptoms_end, to_R!, model)  # Symptoms do not persist after infectious period -> proceed from IS to R
        end
    else                        # Person will be hospitalised
        if dt_exit_H <= state.infectious_end
            schedule!(agent.id, dt_exit_H, to_W!, model)  # Enter ward while still infectious
        else
            schedule!(agent.id, state.infectious_end, to_H!, model)  # Remain home after infectious period before being hospitalised
        end
    end
    if rand() < active_testing_policy.IS  # Schedule test for Covid
        dur_totest = max(2, dur_onset2test(model.params) - 2)  # Time between onset of symptoms and test...at least 2 days
        schedule!(agent.id, dt + Day(dur_totest), test_for_covid!, model)
    end
end

"End of the infectious period."
function to_H!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :H
    state.dt_last_transition = dt
    most_severe_state = state.most_severe_state
    dur_symptomatic   = (state.symptoms_end - state.incubation_end).value
    days_IS           = (dt - state.incubation_end).value  # Days spent with status IS
    if most_severe_state == :H        # Path is: Home->Recovery
        schedule!(agent.id, state.symptoms_end, to_R!, model)
    elseif most_severe_state == :W    # Path is: Home->Ward->Recovery
        dur_home = round(Int, 0.4 * dur_symptomatic) - days_IS   # Split dur_symptomatic: 40% Home, 60% Ward
    elseif most_severe_state == :ICU  # Path is: Home->Ward->ICU->Ward->Recovery
        dur_home = round(Int, 0.25 * dur_symptomatic) - days_IS  # Split dur_symptomatic: 25% Home, 40% Ward, 35% ICU
    elseif most_severe_state == :V    # Path is: Home->Ward->ICU->Vent->ICU->Ward->Recovery
        dur_home = round(Int, 0.2 * dur_symptomatic) - days_IS   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
    else  # most_severe_state == :D   # Path is: Home->Ward->ICU->Vent->deceased
        dur_home = round(Int, 0.2 * dur_symptomatic) - days_IS   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
    end
    if most_severe_state != :H  # Person is hospitalised
        dur_home = max(0, dur_home)
        schedule!(agent.id, dt + Day(dur_home), to_W!, model)  # Proceed from Home to Ward
    end
end

function to_W!(agent::Person{Char, DiseaseProgression}, model, dt)
    state        = agent.state
    prevstate    = state.status
    state.status = :W
    state.dt_last_transition = dt
    state.quarantined = true  # All hospitalised patients are assumed to be quarantined
    rand() < active_testing_policy.W && schedule!(agent.id, dt, test_for_covid!, model)  # Test immediately
    most_severe_state = state.most_severe_state
    dur_symptomatic   = (state.symptoms_end - state.incubation_end).value
    if most_severe_state == :W        # Path is: Home->Ward->Recovery
        schedule!(agent.id, state.symptoms_end, to_R!, model)
    elseif most_severe_state == :ICU  # Path is: Home->Ward->ICU->Ward->Recovery
        dur_ward = round(Int, 0.2 * dur_symptomatic)  # Split dur_symptomatic: 25% Home, 40% Ward, 35% ICU
        if prevstate == :H
            schedule!(agent.id, dt + Day(dur_ward), to_ICU!, model)
        else  # prevstate == :ICU
            schedule!(agent.id, state.symptoms_end, to_R!, model)
        end
    elseif most_severe_state == :V    # Path is: Home->Ward->ICU->Vent->ICU->Ward->Recovery
        dur_ward = round(Int, 0.15 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        if prevstate == :H
            schedule!(agent.id, dt + Day(dur_ward), to_ICU!, model)
        else  # prevstate == :ICU
            schedule!(agent.id, state.symptoms_end, to_R!, model)
        end
    else  # most_severe_state == :D   # Path is: Home->Ward->ICU->Vent->deceased
        dur_ward = round(Int, 0.3 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        schedule!(agent.id, dt + Day(dur_ward), to_ICU!, model)
    end
end

function to_ICU!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    prevstate    = state.status
    state.status = :ICU
    state.dt_last_transition = dt
    most_severe_state = state.most_severe_state
    dur_symptomatic   = (state.symptoms_end - state.incubation_end).value
    if most_severe_state == :ICU     # Path is: Home->Ward->ICU->Ward->Recovery
        dur_icu = round(Int, 0.35 * dur_symptomatic)  # Split dur_symptomatic: 25% Home, 40% Ward, 35% ICU
        schedule!(agent.id, dt + Day(dur_icu), to_W!, model)
    elseif most_severe_state == :V   # Path is: Home->Ward->ICU->Vent->ICU->Ward->Recovery
        dur_icu = round(Int, 0.15 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        if prevstate == :W
            schedule!(agent.id, dt + Day(dur_icu), to_V!, model)
        else  # prevstate == :V
            schedule!(agent.id, dt + Day(dur_icu), to_W!, model)
        end
    else  # most_severe_state == :D  # Path is: Home->Ward->ICU->Vent->deceased
        dur_icu = round(Int, 0.3 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        schedule!(agent.id, dt + Day(dur_icu), to_V!, model)
    end
end

function to_V!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :V
    state.dt_last_transition = dt
    dur_symptomatic = (state.symptoms_end - state.incubation_end).value
    if state.most_severe_state == :V       # Path is: Home->Ward->ICU->Vent->ICU->Ward->Recovery
        dur_vent = round(Int, 0.2 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        schedule!(agent.id, dt + Day(dur_vent), to_ICU!, model)
    else  # most_severe_state == :D  # Path is: Home->Ward->ICU->Vent->deceased
        dur_vent = round(Int, 0.2 * dur_symptomatic)   # Split dur_symptomatic: 20% Home, 30% Ward, 30% ICU, 20% Ventilator
        schedule!(agent.id, dt + Day(dur_vent), to_D!, model)
    end
end

function to_R!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    state.status = :R
    state.dt_last_transition = dt
    state.last_test_date     = dummydate()
    state.last_test_result   = 'n'
    state.quarantined = false
end

to_D!(agent::Person{Char, DiseaseProgression}, model, dt) = agent.state.status = :D

################################################################################
# Infect contacts event

function infect_contacts!(agent::Person{Char, DiseaseProgression}, model, dt)
    state = agent.state
    !(dt >= state.infectious_start && dt <= state.infectious_end) && return  # Person is not infectious
    state.quarantined && return  # A quarantined person cannot infect anyone
    dow       = dayofweek(dt)    # 1 = Monday, ..., 7 = Sunday
    agents    = model.agents
    params    = model.params
    pr_infect = p_infect(params)
    n_susceptible_contacts = 0
    ncontacts = get_contactlist(agent, :household, params)
    n_susceptible_contacts = infect_contactlist!(ncontacts, active_distancing_policy.household, pr_infect, agents, model, dt, n_susceptible_contacts)
    if dow <= 5
        ncontacts = get_contactlist(agent, :school, params)
        n_susceptible_contacts = infect_contactlist!(ncontacts, active_distancing_policy.school,    pr_infect, agents, model, dt, n_susceptible_contacts)
        ncontacts = get_contactlist(agent, :workplace, params)
        n_susceptible_contacts = infect_contactlist!(ncontacts, active_distancing_policy.workplace, pr_infect, agents, model, dt, n_susceptible_contacts)
    end
    ncontacts = get_contactlist(agent, :community, params)
    n_susceptible_contacts = infect_contactlist!(ncontacts, active_distancing_policy.community, pr_infect, agents, model, dt, n_susceptible_contacts)
    ncontacts = get_contactlist(agent, :social, params)
    n_susceptible_contacts = infect_contactlist!(ncontacts, active_distancing_policy.social,    pr_infect, agents, model, dt, n_susceptible_contacts)
    n_susceptible_contacts > 0 && dt < state.infectious_end && schedule!(agent.id, dt + Day(1), infect_contacts!, model)
end

function infect_contactlist!(ncontacts, pr_contact, pr_infect, agents, model, dt, n_susceptible_contacts)
    for i = 1:ncontacts
        contactid = getcontact(i)
        contact   = agents[contactid]
        n_susceptible_contacts = infect_contact!(pr_contact, pr_infect, contact, model, dt, n_susceptible_contacts)
    end
    n_susceptible_contacts
end

function infect_contact!(pr_contact, pr_infect, contact, model, dt, n_susceptible_contacts)
    state = contact.state
    if !(state.quarantined) && pr_contact > 0.0 && rand() <= pr_contact  # Contact between agent and contact occurs
        if state.status == :S && rand() <= pr_infect  # The agent infects the contact
            execute_event!(to_E!, contact, model, dt, metrics)
        #elseif state.status == :R && rand() <= state.p_reinfection
        #    execute_event!(to_E!, contact, model, dt, metrics)
        end
    end
    if state.status == :S
        n_susceptible_contacts += 1
    end
    n_susceptible_contacts
end

################################################################################
# Functions dependent on model parameters; called by event functions

"""
Time between onset of symptoms (start IS) and taking a test.
Drawn from Gamma and constrained to be between 1 and 16 days inclusive, which is about the 95th percentile.
"""
function dur_onset2test(params)
    dur = max(1, round(Int, rand(Gamma(params[:shape_onset2test],  params[:scale_onset2test]))))
    min(dur, 16)
end

"""
Duration of E + IA.
Drawn from LogNormal and constrained to be between 1 and 12 days inclusive, which is about the 95th percentile.
"""
function dur_incubation(params)
    dur = max(1, round(Int, rand(LogNormal(params[:mu_incubation], params[:sigma_incubation]))))
    min(dur, 12)
end

"""
Duration of IA + IS.
Drawn from LogNormal and constrained to be between 1 and 14 days inclusive, which is about the 95th percentile.
"""
function dur_infectious(params)
    dur = max(1, round(Int, rand(LogNormal(params[:mu_infectious], params[:sigma_infectious]))))
    min(dur, 14)
end

"Draw total duration of symtpoms given person will experience symptoms"
function draw_total_duration_of_symptoms(age, most_severe_state::Symbol, params)
    mu = params[:b4] * age + params[:b0]
    if most_severe_state == :W
        mu += params[:b1]
    elseif most_severe_state == :ICU
        mu += params[:b1] + params[:b2]
    elseif most_severe_state == :V
        mu += params[:b1] + params[:b2] + params[:b3]
    end
    d   = LogNormal(mu, params[:sigma_symptoms])
    dur = max(1, round(Int, rand(d)))
    min(dur, 100)
end

"Draw the most severe state that the person will experience"
function draw_most_severe_state(age)
    agegroup_lb = min(80, 10 * div(age, 10))
    d = lb2dist[agegroup_lb]  # Age-specific Categorical distribution
    most_severe_states[rand(d)]
end

p_infect(params) = params[:p_infect]  # Pr(Person infects contact | Person makes contact with contact)

end