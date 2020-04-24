module config

export Config

using YAML

struct Scenario
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
        school     > 0.0 && workplace > 0.0  && error("Cannot make contacts both in school and the work place.")
        new(household, school, workplace, community, social)
    end
end

Scenario(d::Dict) = Scenario(d["household"], d["school"], d["workplace"], d["community"], d["social"])

struct Config
    input_data::Dict{String, String}  # tablename => datafile
    output_directory::String
    initial_state_distribution::Vector{Float64}
    maxtime::Int
    n_social_contacts::Int
    n_community_contacts::Int
    scenarios::Dict{String, Scenario}  # scenario_name => scenario

    function Config(input_data, output_directory, initial_state_distribution, maxtime, n_social_contacts, n_community_contacts, scenarios)
        for (tablename, datafile) in input_data
            !isfile(datafile) && error("Input data file does not exist: $(datafile)")
        end
        !isdir(output_directory) && error("The output directory does not exist: $(output_directory)")
        length(initial_state_distribution) != 8 && error("Initial state distribution does not have length 8")
        maxtime < 0 && error("maxtime is less than 0")
        n_social_contacts    < 0 && error("n_social_contacts must be at least 0")
        n_community_contacts < 0 && error("n_community_contacts must be at least 0")
        new(input_data, output_directory, initial_state_distribution, maxtime, n_social_contacts, n_community_contacts, scenarios)
    end
end

function Config(configfile::String)
    d = YAML.load_file(configfile)
    scenarios = Dict(nm => Scenario(x) for (nm, x) in d["scenarios"])
    Config(d["input_data"], d["output_directory"], d["initial_state_distribution"], d["maxtime"],
           d["n_social_contacts"], d["n_community_contacts"], scenarios)
end

end