module run_model

export runmodel

import Base.run  # Extending this function

using CSV
using DataFrames
using Dates
using Logging

include("abm.jl")  # Depends on: core, config, contacts
using .abm

function runmodel(configfile::String)
    @info "$(now()) Configuring model"
    cfg      = Config(configfile)
    firstday = cfg.firstday
    lastday  = cfg.lastday

    @info "$(now()) Initialising model"
    params = construct_params(cfg.paramsfile)
    model  = init_model(params, cfg)

    @info "$(now()) Initialising metrics"
    initialise_metrics(model.agents)

    @info "$(now()) Initialising output data"
    output  = init_output(metrics, firstday, lastday)  # 1 row for each date-address combination. Requires metrics to be initialised.
    outfile = joinpath(cfg.output_directory, "metrics.csv")

    # Run model
    agents = model.agents
    for r in 1:cfg.nruns
        @info "$(now())    Starting run $(r)"
        reset_model!(model, cfg)
        reset_metrics!(model)
        reset_output!(output)
        i_output = 0  # The number of rows of output that have been populated
        for date in firstday:Day(1):lastday
            model.date = date
            update_policies!(cfg, date, date > firstday)
            apply_forcing!(cfg.forcing, model, date)
            execute_events!(model.schedule, date, agents, model, metrics)
            i_output, i_total = metrics_to_output!(metrics, output, r, date, i_output)  # System at 11:59pm
        end
        CSV.write(outfile, output; delim=',', append=r>1)
    end
    @info "$(now()) Finished. Results written to $(outfile)"
end

################################################################################
# Utils

function construct_params(paramsfile::String)
    tbl = DataFrame(CSV.File(paramsfile))
    Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(tbl.name, tbl.value))
end

end
