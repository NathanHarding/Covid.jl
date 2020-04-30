module Covid

using CSV
using DataFrames
using Dates
using Logging

include(joinpath("abm", "abm.jl"))  # Depends on: core, abm/config, abm/contacts
using .abm

function main(configfile::String)
    @info "$(now()) Configuring model"
    cfg = abm.Config(configfile)

    @info "$(now()) Importing input data"
    indata = import_data(cfg.input_data)

    @info "$(now()) Initialising model"
    params = construct_params(indata["params"])
    model  = abm.init_model(indata, params, cfg)

    @info "$(now()) Initialising output data"
    metrics = abm.init_metrics()
    output  = init_output(metrics, model.maxtime + 1)  # Core function

    # Run scenarios
    for (scenarioname, scenario) in cfg.scenarios
        @info "$(now()) Running $(scenarioname) scenario"
        outfile = joinpath(cfg.output_directory, "$(scenarioname).csv")
        for r in 1:cfg.nruns_per_scenario
            @info "$(now())    Starting run $(r)"
            abm.reset_model!(model)
            abm.reset_metrics!(model, metrics)
            reset_output!(output, r)                # Core function
            run!(model, scenario, metrics, output)  # Core function
            CSV.write(outfile, output; delim=',', append=r>1)
        end
    end
    @info "$(now()) Finished"
end

################################################################################
# Utils

function import_data(tablename2datafile::Dict{String, String})
    result = Dict{String, DataFrame}()
    for (tablename, datafile) in tablename2datafile
        result[tablename] = DataFrame(CSV.File(datafile))
    end
    result
end

function construct_params(params::DataFrame)
    d = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(params.name, params.value))
    dict_to_namedtuple(d)
end

function dict_to_namedtuple(d::Dict{Symbol, V}) where V
    (; zip(Tuple(collect(keys(d))), Tuple(collect(values(d))))...)
end

end
