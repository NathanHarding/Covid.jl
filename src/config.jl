module config

export Config, DistancingPolicy, TestingPolicy, TracingPolicy, QuarantinePolicy

using Dates
using YAML
using CSV
using DataFrames

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
    demographics_datadir::String  # Directory containing saved population data. people = load(demographics_datadir)
    output_directory::String
    firstday::Date  # Simulation starts at 12am
    lastday::Date   # Simulation finishes at 12am
    nruns::Int
    paramsfile::String
    forcing::Dict{Date, Dict{Any,Dict{Symbol, Int}}}            # date => {SA2 => {status => count(status)}}
    t2distancingpolicy::Dict{Date, DistancingPolicy}  # t => distancing_policy
    t2testingpolicy::Dict{Date, TestingPolicy}        # t => testing_policy
    t2tracingpolicy::Dict{Date, TracingPolicy}        # t => tracing_policy
    t2quarantinepolicy::Dict{Date, QuarantinePolicy}  # t => quarantine_policy
    cumsum_population::DataFrame

    function Config(demographics_datadir, outdir, firstday, lastday, nruns, paramsfile, forcing, t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy,cumsum_population)
        !isdir(dirname(demographics_datadir)) && error("The directory containing the output directory does not exist: $(dirname(demographics_datadir))")
        !isdir(dirname(outdir)) && error("The directory containing the output directory does not exist: $(dirname(outdir))")
        if !isdir(outdir)
            mkdir(outdir)
        end
        firstday > lastday && error("First day is after last day")
        nruns < 1 && error("nruns is less than 1")
        paramsfile_attempt = constructpath(paramsfile, '/')
        paramsfile = isfile(paramsfile_attempt) ? paramsfile_attempt : constructpath(joinpath(pwd(), paramsfile), '/')
        !isfile(paramsfile) && error("The file containing model parameters does not exist: $(paramsfile)")
        registered_statuses = Set([:S, :E, :IA, :IS, :W, :ICU, :V, :R, :D])
        for (dt, status2n) in forcing
            for (SA2,seed) in status2n
                for (status, n) in seed
                    !in(status, registered_statuses) && error("Unknown status $(status)")
                    n < 0 && error("Initial count for status $(status) is $(n). Must be non-negative.")
                end
            end
        end
        new(demographics_datadir, outdir, firstday, lastday, nruns, paramsfile, forcing, t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy,cumsum_population)
    end
end

Config(configfile::String) = Config(YAML.load_file(configfile))

function Config(d::Dict)
    datadir_attempt    = constructpath(d["demographics_datadir"],  '/')
    demographics_datadir = isdir(datadir_attempt) ? datadir_attempt : constructpath(joinpath(pwd(), d["demographics_datadir"]), '/')
    outdir_attempt     = constructpath(d["output_directory"], '/')
    outdir             = isdir(outdir_attempt) ? outdir_attempt : constructpath(joinpath(pwd(), d["output_directory"]), '/')
    firstday           = Date(d["firstday"])
    lastday            = Date(d["lastday"])
    nruns              = d["nruns"]
    paramsfile         = d["params"]
    forcing            = construct_forcing(d["forcing"])
    t2distancingpolicy = isnothing(d["distancing_policy"]) ? Dict{Date, DistancingPolicy}() : Dict(Date(dt) => DistancingPolicy(policy) for (dt, policy) in d["distancing_policy"])
    t2testingpolicy    = isnothing(d["testing_policy"])    ? Dict{Date, TestingPolicy}()    : Dict(Date(dt) => TestingPolicy(policy)    for (dt, policy) in d["testing_policy"])
    t2tracingpolicy    = isnothing(d["tracing_policy"])    ? Dict{Date, TracingPolicy}()    : Dict(Date(dt) => TracingPolicy(policy)    for (dt, policy) in d["tracing_policy"])
    t2quarantinepolicy = isnothing(d["quarantine_policy"]) ? Dict{Date, QuarantinePolicy}() : Dict(Date(dt) => QuarantinePolicy(policy) for (dt, policy) in d["quarantine_policy"])
    cumsum_population  = DataFrame(CSV.File(d["cumsum_population"]))
    Config(demographics_datadir, outdir, firstday, lastday, nruns, paramsfile, forcing, t2distancingpolicy, t2testingpolicy, t2tracingpolicy, t2quarantinepolicy,cumsum_population)
end

"Constructs a valid file or directory path by ensuring the correct separator for the operating system."
function constructpath(s::String, sep::Char)
    parts = split(s, sep)
    normpath(joinpath(parts...))
end

function construct_forcing(d::Dict)
    result = Dict{Date, Dict{Any,Dict{Symbol, Int}}}()
    for (dt, status2n) in d
        result[Date(dt)] = Dict{Any,Dict{Symbol, Int}}()
        for (SA2,seed) in status2n
            result[Date(dt)][SA2] = Dict(Symbol(status) => n for (status, n) in seed)
        end
    end
    result
end

end