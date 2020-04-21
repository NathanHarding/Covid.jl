module run

export main

using CSV

using ..config
using ..utils
using ..core
using ..abm

function main(configfile::String)
    cfg     = Config(configfile)
    indata  = utils.import_data(cfg.input_data)
    params  = indata["params"]
    params  = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(params.name, params.value))
    params  = utils.dict_to_namedtuple(params)
    model   = init_model(indata, params, cfg.maxtime, cfg.initial_state_distribution)
    outdata = run!(model)
    CSV.write(cfg.output_data, outdata; delim='\t')
end

end