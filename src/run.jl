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
    params = indata["params"]
    params = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(params.name, params.value))
    params = utils.dict_to_namedtuple(params)
    model  = abm.init_model(indata, params, cfg)

    # Run scenarios
    metrics = abm.init_metrics()
    output  = init_output(metrics, model.maxtime + 1)
    for (scenarioname, scenario) in cfg.scenarios
        @info "$(now()) Running $(scenarioname) scenario"
        outfile = joinpath(cfg.output_directory, "$(scenarioname).csv")
        for r in 1:cfg.nruns_per_scenario
            abm.reset_model!(model)
            abm.reset_metrics!(model, metrics)
            reset_output!(output)
            t0 = now()
            run!(model, scenario, r, metrics, output, abm.unfit!, abm.fit!)
            t1 = now()
            ms = (t1 - t0).value
            s  = round(0.001*ms; digits=3)
            @info "    Run $(r) took $(s) seconds"
            CSV.write(outfile, output; delim=',', append=r>1)
        end
    end
    @info "$(now()) Finished"
end

end