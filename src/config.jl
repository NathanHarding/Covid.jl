module config

export Config, DistancingRegime, TestingRegime

using YAML

################################################################################
"Updated over time"
mutable struct DistancingRegime
    household::Float64
    school::Float64
    workplace::Float64
    community::Float64
    social::Float64

    function DistancingRegime(household, school, workplace, community, social)
        (household < 0.0 || household > 1.0) && error("Pr(Contact within household) must be between 0 and 1")
        (school    < 0.0 || school    > 1.0) && error("Pr(Contact within school) must be between 0 and 1")
        (workplace < 0.0 || workplace > 1.0) && error("Pr(Contact within work place) must be between 0 and 1")
        (community < 0.0 || community > 1.0) && error("Pr(Contact within community) must be between 0 and 1")
        (social    < 0.0 || social    > 1.0) && error("Pr(Contact within social network) must be between 0 and 1")
        new(household, school, workplace, community, social)
    end
end

DistancingRegime(d::Dict) = DistancingRegime(d["household"], d["school"], d["workplace"], d["community"], d["social"])

################################################################################
"Pr(Test person | Status). People tested under this regime are NOT contacts of known cases."
mutable struct TestingRegime
   S::Float64
   E::Float64
   IA::Float64
   IS::Float64
   W::Float64
   ICU::Float64
   V::Float64

   function TestingRegime(S, E, IA, IS, W, ICU, V)
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

function TestingRegime(d::Dict)
    result = TestingRegime(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for (status, prob) in d
        (prob < 0.0 || prob > 1.0) && error("TestingRegime has an invalid probability. status = $(status); prob = $(prob)")
        setfield!(result, Symbol(status), prob)
    end
    result    
end

################################################################################
struct Config
    datadir::String
    input_data::Dict{String, String}  # tablename => datafile
    initial_status_counts::Dict{Symbol, Int}  # status => count(status)
    maxtime::Int
    nruns::Int
    t2distancingregime::Dict{Int, DistancingRegime}  # t => distancing regime
    t2testingregime::Dict{Int, TestingRegime}        # t => testing_regime

    function Config(datadir, input_data, initial_status_counts, maxtime, nruns, t2distancingregime, t2testingregime)
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
        maxtime < 0 && error("maxtime is less than 0")
        nruns   < 1 && error("nruns is less than 1")
        new(datadir, input_data, initial_status_counts, maxtime, nruns, t2distancingregime, t2testingregime)
    end
end

function Config(configfile::String)
    d = YAML.load_file(configfile)
    status0            = Dict(Symbol(k) => v for (k, v) in d["initial_status_counts"])
    t2distancingregime = Dict(t => DistancingRegime(regime) for (t, regime) in d["distancing_regime"])
    t2testingregime    = Dict(t => TestingRegime(regime)    for (t, regime) in d["testing_regime"])
    Config(d["datadir"], d["input_data"], status0, d["maxtime"], d["nruns"], t2distancingregime, t2testingregime)
end

end