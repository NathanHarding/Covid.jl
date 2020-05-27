module trainmodel

function train!(configfile::String)
    @info "$(now()) Configuring model"
    cfg = Config(configfile)

    @info "$(now()) Importing input data"
    indata = import_data(cfg.datadir, cfg.input_data)

    @info "$(now()) Initialising output data"
    output  = init_output(metrics, cfg.firstday, cfg.lastday)  # 1 row for each date
    outfile = joinpath(cfg.datadir, "output", "metrics.csv")

    @info "$(now()) Initialising model"
    params = construct_params(indata["params"])
    model  = init_model(indata, params, cfg)

    # Run model
    firstday = cfg.firstday
    lastday  = cfg.lastday
    agents   = model.agents
    for r in 1:cfg.nruns
        reset_model!(model, cfg)
        reset_metrics!(model)
        reset_output!(output, r)
        for date in firstday:Day(1):lastday
            model.date = date
            metrics_to_output!(metrics, output, date)  # System as of 12am on date
            date == lastday && break
            update_policies!(cfg, date)
            execute_events!(model.schedule[date], agents, model, date, metrics)
        end
    end
    @info "$(now()) Finished"
end


end