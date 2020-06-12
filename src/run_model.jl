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

    @info "$(now()) Importing input data"
    indata = import_data(cfg.datadir, cfg.input_data)

    @info "$(now()) Initialising output data"
    output  = init_output(metrics, cfg.firstday, cfg.lastday)  # 1 row for each date
    outfile = joinpath(cfg.datadir, "output", "metrics.csv")

    @info "$(now()) Initialising model"
    params = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(indata["params"].name, indata["params"].value))
    model  = init_model(indata, params, cfg)

    # Run model
    firstday = cfg.firstday
    lastday  = cfg.lastday
    agents   = model.agents
    for r in 1:cfg.nruns
        @info "$(now())    Starting run $(r)"
        reset_model!(model, cfg)
        reset_metrics!(model)
        reset_output!(output, r)
        for date in firstday:Day(1):lastday
            model.date = date
            metrics_to_output!(metrics, output, date)  # System as of 12am on date
            date == lastday && break
            update_policies!(cfg, date)
            apply_forcing!(cfg.forcing, model, date)
            execute_events!(model.schedule[date], agents, model, date, metrics)
        end
        CSV.write(outfile, output; delim=',', append=r>1)
        GC.gc()
    end
    @info "$(now()) Finished. Results written to $(outfile)"
end

################################################################################
# Utils

function import_data(datadir::String, tablename2datafile::Dict{String, String})
    result = Dict{String, DataFrame}()
    for (tablename, datafile) in tablename2datafile
        filename = joinpath(datadir, "input", datafile)
        result[tablename] = DataFrame(CSV.File(filename))
    end
    result
end

end
