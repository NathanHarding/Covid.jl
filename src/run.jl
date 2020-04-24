module run

export main

using CSV
using Dates
using Logging

using ..config
using ..utils
using ..core
using ..abm

function main(configfile::String)
    @info "$(now()) Configuring model"
    cfg = Config(configfile)

    @info "$(now()) Importing input data"
    indata = utils.import_data(cfg.input_data)

    @info "$(now()) Initialising model"
    params  = indata["params"]
    params  = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(params.name, params.value))
    params  = utils.dict_to_namedtuple(params)
    model   = init_model(indata, params, cfg)
    status0 = [agent.status for agent in model.agents]  # Initial status for each agent

    # Run scenarios
    i = 0
    for (scenarioname, scenario) in cfg.scenarios
        @info "$(now()) Running $(scenarioname) scenario"
        i += 1
        if i >= 2
            reset_model!(model, status0)
        end
        outdata = run!(model, scenario)
        outfile = joinpath(cfg.output_directory, "$(scenarioname).csv")
        CSV.write(outfile, outdata; delim=',')
        @info "$(now())     Results written to $(outfile)"
    end
    @info "$(now()) Finished"
end

function reset_model!(model, status0)
    agents = model.agents
    n = length(agents)
    for i = 1:n
        agents[i].status = status0[i]
    end
end

end