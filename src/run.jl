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
    cfg     = Config(configfile)

    @info "$(now()) Importing input data"
    indata  = utils.import_data(cfg.input_data)
    params  = indata["params"]
    params  = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(params.name, params.value))
    params  = utils.dict_to_namedtuple(params)

    @info "$(now()) Initialising model"
    model   = init_model(indata, params, cfg.maxtime, cfg.initial_state_distribution)

    @info "$(now()) Running model"
    outdata = run!(model)

    @info "$(now()) Writing output data to disk"
    CSV.write(cfg.output_data, outdata; delim='\t')
end

end