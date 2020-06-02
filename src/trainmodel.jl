module trainmodel

export train!

using CSV
using DataFrames
using Dates
using Logging
using Optim
using YAML

include("abm.jl")  # Depends on: core, config, contacts
using .abm

function train!(configfile::String)
    @info "$(now()) Configuring model"
    d   = YAML.load_file(configfile)
    cfg = Config(d)
    unknowns, n_unknowns = construct_unknowns(d["unknowns"])

    @info "$(now()) Importing input data"
    indata = import_data(cfg.datadir, cfg.input_data)

    @info "$(now()) Preparing training data"
    y = prepare_training_data(cfg.datadir, d)  # date => (colname1=val1, ...)

    @info "$(now()) Initialising model"
    params = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(indata["params"].name, indata["params"].value))
    model  = init_model(indata, params, cfg)

    @info "$(now()) Training model"
    theta0 = fill(0.0, n_unknowns)
    params_and_policies_to_theta!(theta0, unknowns, cfg, model.params)
    loss(theta) = lossfunc(theta, y, model, cfg, metrics, unknowns)
    opts   = Optim.Options(; Dict{Symbol, Any}(Symbol(k) => v for (k, v) in d["solver_options"])...)
    result = optimize(loss, theta0, NelderMead(), opts)

    @info "$(now()) Extracting result"
    theta = result.minimizer
    theta_to_params_and_policies!(theta, unknowns, cfg, model.params)
for x in theta
    println("$(logit_to_prob(x))")
end
    write_theta(theta, unknowns, cfg, model.params)
    @info "$(now()) Finished"
end

################################################################################
# Construct unknowns

function construct_unknowns(d)
    result     = Dict{Symbol, Any}()
    n_unknowns = format_param_unknowns!(result, 0, d)
    n_unknowns = format_policy_unknowns!(result, n_unknowns, d, "distancing_policy")
    n_unknowns = format_policy_unknowns!(result, n_unknowns, d, "testing_policy")
    n_unknowns = format_policy_unknowns!(result, n_unknowns, d, "tracing_policy")
    n_unknowns = format_policy_unknowns!(result, n_unknowns, d, "quarantine_policy")
    result, n_unknowns
end

function format_param_unknowns!(result, n_unknowns, d)
    if haskey(d, "params")
        result[:params] = Symbol[]
        params = d["params"]
        if params isa String
            n_unknowns += 1
            push!(result[:params], Symbol(params))
        elseif params isa Vector{String}
            for x in params
                n_unknowns += 1
                push!(result[:params], Symbol(x))
            end
        else
            error("Type of unknown params must be String or Vector{String}")
        end
    end
    n_unknowns
end

function format_policy_unknowns!(result, n_unknowns, d, policyname_str::String)
    !haskey(d, policyname_str) && return n_unknowns
    d2 = d[policyname_str]
    isempty(d2) && return n_unknowns
    policyname = Symbol(replace(policyname_str, "_" => ""))  # Example: "distancing_policy" => :distancingpolicy
    result[policyname] = Pair{Date, Vector{Symbol}}[]  # Using a Vector so that order is preserved
    for (dt, flds) in d2
        newvec = Symbol[]
        if flds isa String
            n_unknowns += 1
            push!(newvec, Symbol(flds))
        elseif flds isa Vector{String}
            for fld in flds
                n_unknowns += 1
                push!(newvec, Symbol(fld))
            end
        else
            error("Type of unknown policy elements must be String or Vector{String}")
        end
        push!(result[policyname], Date(dt) => newvec)
    end
    sort!(result[policyname], by = (x) -> x[1])  # Sort by date, earliest to latest. Not necessary but easier to keep track of.
    n_unknowns
end

################################################################################
# Converting from params and policies to theta, and back again.

"Modified: theta"
function params_and_policies_to_theta!(theta, unknowns, cfg, params)
    i = haskey(unknowns, :params)           ? params_to_theta!(theta, 0, unknowns[:params], params) : 0
    i = haskey(unknowns, :distancingpolicy) ? policy_to_theta!(theta, i, unknowns[:distancingpolicy], cfg.t2distancingpolicy) : i
    i = haskey(unknowns, :testingpolicy)    ? policy_to_theta!(theta, i, unknowns[:testingpolicy],    cfg.t2testingpolicy)    : i
    i = haskey(unknowns, :tracingpolicy)    ? policy_to_theta!(theta, i, unknowns[:tracingpolicy],    cfg.t2tracingpolicy)    : i
    i = haskey(unknowns, :quarantinepolicy) ? policy_to_theta!(theta, i, unknowns[:quarantinepolicy], cfg.t2quarantinepolicy) : i
end

"Modified: params, cfg."
function theta_to_params_and_policies!(theta, unknowns, cfg, params)
    i = haskey(unknowns, :params)           ? theta_to_params!(theta, 0, unknowns[:params], params) : 0
    i = haskey(unknowns, :distancingpolicy) ? theta_to_policy!(theta, i, unknowns[:distancingpolicy], cfg.t2distancingpolicy) : i
    i = haskey(unknowns, :testingpolicy)    ? theta_to_policy!(theta, i, unknowns[:testingpolicy],    cfg.t2testingpolicy)    : i
    i = haskey(unknowns, :tracingpolicy)    ? theta_to_policy!(theta, i, unknowns[:tracingpolicy],    cfg.t2tracingpolicy)    : i
    i = haskey(unknowns, :quarantinepolicy) ? theta_to_policy!(theta, i, unknowns[:quarantinepolicy], cfg.t2quarantinepolicy) : i
end

function params_to_theta!(theta, i, unknowns::Vector{Symbol}, params)
    for nm in unknowns
        i += 1
        theta[i] = prob_to_logit(params[nm])
    end
    i
end

function theta_to_params!(theta, i, unknowns::Vector{Symbol}, params::Dict{Symbol, Float64}, )
    for nm in unknowns
        i += 1
        params[nm] = logit_to_prob(theta[i])
    end
    i
end

function policy_to_theta!(theta, i, unknowns::Vector{Pair{Date, Vector{Symbol}}}, dt2policy)
    for (dt, flds) in unknowns
        policy = dt2policy[dt]
        for fld in flds
            i += 1
            theta[i] = prob_to_logit(getfield(policy, fld))
        end
    end
    i
end

function theta_to_policy!(theta, i, unknowns::Vector{Pair{Date, Vector{Symbol}}}, dt2policy)
    for (dt, flds) in unknowns
        policy = dt2policy[dt]
        for fld in flds
            i += 1
            setfield!(policy, fld, logit_to_prob(theta[i]))
        end
    end
    i
end

################################################################################
# Loss function

function lossfunc(theta::Vector{Float64}, y, model, cfg, metrics, unknowns)
    LL = 0.0
    nruns    = cfg.nruns
    firstday = cfg.firstday
    lastday  = cfg.lastday
    agents   = model.agents
    theta_to_params_and_policies!(theta, unknowns, cfg, model.params)
    for r in 1:nruns
        reset_model!(model, cfg)
        reset_metrics!(model)
        for date in firstday:Day(1):lastday
            model.date = date
            LL += loglikelihood(y[date][:positives], metrics[:positives])  # System as of 12am on date
            date == lastday && break
            update_policies!(cfg, date)
            execute_events!(model.schedule[date], agents, model, date, metrics)
        end
    end
    ndays = (lastday - firstday).value + 1
    -LL / (ndays * nruns)
end

function loglikelihood(y, yhat)
    yhat <= 0.01 && return 0.0
    y * log(yhat) - yhat  # LL(Y ~ Poisson(yhat)) without constant term
end

################################################################################
# Utils

logit_to_prob(b) = 1.0 / (1.0 + exp(-b))
prob_to_logit(p) = log(p / (1.0 - p))

function import_data(datadir::String, tablename2datafile::Dict{String, String})
    result = Dict{String, DataFrame}()
    for (tablename, datafile) in tablename2datafile
        filename = joinpath(datadir, "input", datafile)
        result[tablename] = DataFrame(CSV.File(filename))
    end
    result
end

"Returns: Dict{Date, NamedTuple}.  date => (colname1=val1, ...)."
function prepare_training_data(datadir, d)
    y = DataFrame(CSV.File(joinpath(datadir, "input", d["training_data"])))  # Columns: date, newpositives, positives
    sort!(y, (:date,))
    prevdate = nothing
    result   = nothing
    n = size(y, 1)
    for i = 1:n
        dt = Date(y[i, :date])
        nt = copy(y[i, :])  # Row as NamedTuple
        if isnothing(prevdate)
            T = typeof(nt)
            result     = Dict{Date, T}()
            result[dt] = nt
            prevdate   = dt
        else
            for t in (prevdate + Day(1)):Day(1):(dt - Day(1))
                result[t] = result[prevdate]  # Only valid for cumulative data
            end
            result[dt] = nt
        end
    end
    result
end

function write_theta(theta, unknowns, cfg, params)
    result = DataFrame(name=String[], value=Float64[])
    n = size(theta, 1)
    for i = 1:n
        nm  = Symbol("x$(i)")  # TODO: Fix this using unknowns
        p   = logit_to_prob(theta[i])
        row = (name="", value=p)
        push!(result, row)
    end
    outfile = joinpath(cfg.datadir, "output", "trained_params.tsv")
    CSV.write(outfile, result; delim='\t')
end

end