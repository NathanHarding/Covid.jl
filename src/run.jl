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
    for (scenarioname, scenario) in cfg.scenarios
        @info "$(now()) Running $(scenarioname) scenario"
        outfile = joinpath(cfg.output_directory, "$(scenarioname).csv")
        for r in 1:cfg.nruns_per_scenario
            reset_model!(model, status0)
            t0 = now()
            outdata = run!(model, scenario, r)
            t1 = now()
            ms = (t1 - t0).value
            s  = round(0.001*ms; digits=3)
            @info "    Run $(r) took $(s) seconds"
            CSV.write(outfile, outdata; delim=',', append=r>1)
#            if r == 1
#                CSV.write(outfile, outdata; delim=',')
#            else
#                CSV.write(outfile, outdata; delim=',', append=true)
#            end
        end
    end
    @info "$(now()) Finished"
end

function reset_model!(model, status0)
    agents = model.agents
    n = length(agents)
    for i = 1:n
        agents[i].status = status0[i]
    end
    empty!(model.schedule0)
    for x in model.schedule
        empty!(x)
    end
end

end