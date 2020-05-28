module config

export Config, DistancingPolicy, TestingPolicy, TracingPolicy, QuarantinePolicy

using Dates
using YAML

################################################################################
"Updated over time"
mutable struct DistancingPolicy
    household::Float64
    school::Float64
    workplace::Float64
    community::Float64
    social::Float64

    function DistancingPolicy(household, school, workplace, community, social)
        (household < 0.0 || household > 1.0) && error("Pr(Contact within household) must be between 0 and 1")
        (school    < 0.0 || school    > 1.0) && error("Pr(Contact within school) must be between 0 and 1")
        (workplace < 0.0 || workplace > 1.0) && error("Pr(Contact within work place) must be between 0 and 1")
        (community < 0.0 || community > 1.0) && error("Pr(Contact within community) must be between 0 and 1")
        (social    < 0.0 || social    > 1.0) && error("Pr(Contact within social network) must be between 0 and 1")
        new(household, school, workplace, community, social)
    end
end

DistancingPolicy(d::Dict) = DistancingPolicy(d["household"], d["school"], d["workplace"], d["community"], d["social"])

################################################################################
"Pr(Test | Person is not a contact of a known case, status)"
mutable struct TestingPolicy
   S::Float64
   E::Float64
   IA::Float64
   IS::Float64
   W::Float64
   ICU::Float64
   V::Float64

   function TestingPolicy(S, E, IA, IS, W, ICU, V)
       (S   < 0.0 || S   > 1.0) && error("Pr(Test Susceptible person) must be between 0 and 1")
       (E   < 0.0 || E   > 1.0) && error("Pr(Test Exposed person) must be between 0 and 1")
       (IA  < 0.0 || IA  > 1.0) && error("Pr(Test Asymptomatic case) must be between 0 and 1")
       (IS  < 0.0 || IS  > 1.0) && error("Pr(Test Symptomatic case) must be between 0 and 1")
       (W   < 0.0 || W   > 1.0) && error("Pr(Test Ward-bed case) must be between 0 and 1")
       (ICU < 0.0 || ICU > 1.0) && error("Pr(Test ICU unventilated case) must be between 0 and 1")
       (V   < 0.0 || V   > 1.0) && error("Pr(Test Ventilated case) must be between 0 and 1")
       new(S, E, IA, IS, W, ICU, V)
   end
end

function TestingPolicy(d::Dict)
    result = TestingPolicy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for (status, prob) in d
        (prob < 0.0 || prob > 1.0) && error("TestingPolicy has an invalid probability. status = $(status); prob = $(prob)")
        setfield!(result, Symbol(status), prob)
    end
    result    
end

################################################################################
"""
Pr(Test | Person is a contact of a known case, status)

We needn't consider hospitalised symptomatic cases because we assume they are all tested upon admission.
Regarding status, we need only consider whether the person is symptomatic (status is IS) or asymptomatic (status is S, E, IA or R).
"""
mutable struct TracingPolicy
    household::NamedTuple{(:asymptomatic, :symptomatic), Tuple{Float64, Float64}}
    school::NamedTuple{(:asymptomatic, :symptomatic), Tuple{Float64, Float64}}
    workplace::NamedTuple{(:asymptomatic, :symptomatic), Tuple{Float64, Float64}}
    community::NamedTuple{(:asymptomatic, :symptomatic), Tuple{Float64, Float64}}
    social::NamedTuple{(:asymptomatic, :symptomatic), Tuple{Float64, Float64}}

    function TracingPolicy(household, school, workplace, community, social)
        args = [household, school, workplace, community, social]
        flds = (:asymptomatic, :symptomatic)
        for arg in args
            for fld in flds
               p = getfield(arg, fld)
               article = fld == :asymptomatic ? "an" : "a"
               (p < 0.0 || p > 1.0) && error("Pr(Test | Person is $(article) $(fld) $(arg) contact) must be between 0 and 1")
            end
        end
        new(household, school, workplace, community, social)
    end
end

function TracingPolicy(d::Dict)
    household = (asymptomatic=d["household"]["symptomatic"], symptomatic=d["household"]["symptomatic"])
    school    = (asymptomatic=d["school"]["symptomatic"],    symptomatic=d["school"]["symptomatic"])
    workplace = (asymptomatic=d["workplace"]["symptomatic"], symptomatic=d["workplace"]["symptomatic"])
    community = (asymptomatic=d["community"]["symptomatic"], symptomatic=d["community"]["symptomatic"])
    social    = (asymptomatic=d["social"]["symptomatic"],    symptomatic=d["social"]["symptomatic"])
    TracingPolicy(household, school, workplace, community, social)    
end

################################################################################
"""
The quarantine policy specifies how many days people are quarantined, together with the level of compliance (expressed as a probability).
People who are awaiting test results are quarantined until the result is available, which is currently 2 days after the test date.
People who test positive are quarantined for X days after onset of symptoms if they are symptomatic (status=IS), or X days after the test date if asymptomatic.
People known to be in recent contact with a known case are quarantined for X days.
"""
mutable struct QuarantinePolicy
    awaiting_test_result::NamedTuple{(:days, :compliance), Tuple{Int, Float64}}
    tested_positive::NamedTuple{(:days, :compliance), Tuple{Int, Float64}}
    case_contacts::Dict{Symbol, NamedTuple{(:days, :compliance), Tuple{Int, Float64}}}  # Keys are: household, school, workplace, community, social

    function QuarantinePolicy(awaiting_test_result, tested_positive, case_contacts)
        check_quarantine_condition(awaiting_test_result)
        check_quarantine_condition(tested_positive)
        registered_keys   = [:household, :school, :workplace, :community, :social]
        new_case_contacts = Dict{Symbol, NamedTuple{(:days, :compliance), Tuple{Int, Float64}}}()
        for k in registered_keys
            !haskey(case_contacts, k) && continue
            v = case_contacts[k]
            check_quarantine_condition(v)
            new_case_contacts[k] = v
            delete!(case_contacts, k)
        end
        !isempty(case_contacts) && error("case_contacts has invalid keys: $(sort!(collect(keys(case_contacts))))")
        new(awaiting_test_result, tested_positive, new_case_contacts)
    end
end

function QuarantinePolicy(d::Dict)
    awaiting_test_result = construct_quarantine_condition(d["awaiting_test_result"])
    tested_positive      = construct_quarantine_condition(d["tested_positive"])
    case_contacts        = Dict(Symbol(k) => construct_quarantine_condition(v) for (k, v) in d["case_contacts"])
    QuarantinePolicy(awaiting_test_result, tested_positive, case_contacts)
end

construct_quarantine_condition(d::Dict) = (days=d["days"], compliance=d["compliance"])

function check_quarantine_condition(x::NamedTuple{(:days, :compliance), Tuple{Int, Float64}})
    x.days < 0 && error("People cannot be quarantined for a negative number of days")
    (x.compliance < 0.0 || x.compliance > 1.0) && error("Quarantine compliance must be between 0 and 1 inclusive.")    
end

################################################################################
struct Config
    datadir::String
    input_data::Dict{String, String}  # tablename => datafile
    initial_status_counts::Dict{Symbol, Int}  # status => count(status)
    firstday::Date  # Simulation starts at 12am
    lastday::Date   # Simulation finishes at 12am
    nruns::Int
    t2distancingpolicy::Dict{Date, DistancingPolicy}  # t => distancing_policy
    t2testingpolicy::Dict{Date, TestingPolicy}        # t => testing_policy
    t2tracingpolicy::Dict{Date, TracingPolicy}        # t => tracing_policy
    t2quarantinepolicy::Dict{Date, QuarantinePolicy}  # t => quarantine_policy

    function Config(datadir, input_data, initial_status_counts, firstday, lastday, nruns, t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy)
        !isdir(datadir) && error("The data directory does not exist: $(datadir)")
        for (tablename, datafile) in input_data
            filename = joinpath(datadir, "input", datafile)
            !isfile(filename) && error("Input data file does not exist: $(filename)")
        end
        registered_statuses = Set([:S, :E, :IA, :IS, :W, :ICU, :V, :R, :D])
        for (status, n) in initial_status_counts
            !in(status, registered_statuses) && error("Unknown status $(status)")
            n < 0 && error("Initial count for status $(status) is $(n). Must be non-negative.")
        end
        firstday > lastday && error("First day is after last day")
        nruns < 1 && error("nruns is less than 1")
        new(datadir, input_data, initial_status_counts, firstday, lastday, nruns, t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy)
    end
end

function Config(configfile::String)
    d = YAML.load_file(configfile)
    status0            = Dict(Symbol(k) => v for (k, v) in d["initial_status_counts"])
    firstday           = Date(d["firstday"])
    lastday            = Date(d["lastday"])
    t2distancingpolicy = Dict(Date(dt) => DistancingPolicy(policy) for (dt, policy) in d["distancing_policy"])
    t2testingpolicy    = Dict(Date(dt) => TestingPolicy(policy)    for (dt, policy) in d["testing_policy"])
    t2tracingpolicy    = Dict(Date(dt) => TracingPolicy(policy)    for (dt, policy) in d["tracing_policy"])
    t2quarantinepolicy = Dict(Date(dt) => QuarantinePolicy(policy) for (dt, policy) in d["quarantine_policy"])
    Config(d["datadir"], d["input_data"], status0, firstday, lastday, d["nruns"], t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy)
end

end