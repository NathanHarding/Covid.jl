module config

export Config, Scenario

using YAML

mutable struct Scenario
    household::Float64
    school::Float64
    workplace::Float64
    community::Float64
    social::Float64

    function Scenario(household, school, workplace, community, social)
        (household < 0.0 || household > 1.0) && error("Pr(Contact within household) must be between 0 and 1")
        (school    < 0.0 || school    > 1.0) && error("Pr(Contact within school) must be between 0 and 1")
        (workplace < 0.0 || workplace > 1.0) && error("Pr(Contact within work place) must be between 0 and 1")
        (community < 0.0 || community > 1.0) && error("Pr(Contact within community) must be between 0 and 1")
        (social    < 0.0 || social    > 1.0) && error("Pr(Contact within social network) must be between 0 and 1")
        new(household, school, workplace, community, social)
    end
end

Scenario(d::Dict) = Scenario(d["household"], d["school"], d["workplace"], d["community"], d["social"])

struct Config
    datadir::String
    input_data::Dict{String, String}  # tablename => datafile
    initial_status_counts::Dict{Symbol, Int}  # status => count(status)
    maxtime::Int
    nruns_per_scenario::Int
    scenarios::Dict{String, Dict{Int, Scenario}}  # scenario_name => t => scenario

    function Config(datadir, input_data, initial_status_counts, maxtime, nruns_per_scenario, scenarios)
        !isdir(datadir) && error("The data directory does not exist: $(datadir)")
        for (tablename, datafile) in input_data
            filename = joinpath(datadir, "input", datafile)
            !isfile(filename) && error("Input data file does not exist: $(filename)")
        end
        status0 = Dict(Symbol(k) => v for (k, v) in initial_status_counts)
        maxtime < 0 && error("maxtime is less than 0")
        nruns_per_scenario < 1 && error("nruns_per_scenario is less than 1")
        new(datadir, input_data, status0, maxtime, nruns_per_scenario, scenarios)
    end
end

function Config(configfile::String)
    d = YAML.load_file(configfile)
    scenarios = Dict{String, Dict{Int, Scenario}}()
    for (nm, t2scenario) in d["scenarios"]
        scenarios[nm] = Dict{Int, Scenario}()
        for (t, scenario) in t2scenario
            scenarios[nm][t] = Scenario(scenario)
        end
    end
    Config(d["datadir"], d["input_data"], d["initial_status_counts"], d["maxtime"], d["nruns_per_scenario"], scenarios)
end

end