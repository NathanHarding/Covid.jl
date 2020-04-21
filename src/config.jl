module config

export Config

using YAML

struct Config
    input_data::Dict{String, String}  # tablename => datafile
    output_data::String
    initial_state_distribution::Vector{Float64}
    maxtime::Int

    function Config(input_data, output_data, initial_state_distribution, maxtime)
        for (tablename, datafile) in input_data
            !isfile(datafile) && error("Input data file does not exist: $(datafile)")
        end
        !isdir(dirname(output_data)) && error("The directory containing the output data file does not exist: $(dirname(output_data))")
        length(initial_state_distribution) != 8 && error("Initial state distribution does not have length 8")
        maxtime < 0 && error("maxtime is less than 0")
        new(input_data, output_data, initial_state_distribution, maxtime)
    end
end

function Config(configfile::String)
    d = YAML.load_file(configfile)
    Config(d["input_data"], d["output_data"], d["initial_state_distribution"], d["maxtime"])
end

end