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
    cfg = Config(configfile)

    @info "$(now()) Initialising model"
    params = construct_params(cfg.paramsfile)
    model  = init_model(params, cfg)

    @info "$(now()) Initialising metrics"
    initialise_metrics(model.agents)

    @info "$(now()) Initialising output data"
    output  = init_output(metrics, cfg.firstday, cfg.lastday)  # 1 row for each date-address combination. Requires metrics to be initialised.
    outfile = joinpath(cfg.output_directory, "metrics.csv")

    # Run model
    firstday = cfg.firstday
    lastday  = cfg.lastday
    agents   = model.agents
    for r in 1:cfg.nruns
        @info "$(now())    Starting run $(r)"
        reset_model!(model, cfg)
        reset_metrics!(model)
        reset_output!(output)
        for date in firstday:Day(1):lastday
            model.date = date
            metrics_to_output!(metrics, output, r, date)  # System as of 12am on date
            date == lastday && break
            update_policies!(cfg, date)
            apply_forcing!(cfg.forcing, model, date)
            execute_events!(model.schedule[date], agents, model, date, metrics)
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
