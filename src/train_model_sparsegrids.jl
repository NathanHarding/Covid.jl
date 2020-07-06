module train_model

export trainmodel

using CSV
using DataFrames
using Dates
using Distributions
using FastGaussQuadrature  # gausslegendre
using LinearAlgebra  # dot
using Logging
using SparseGrids
using YAML

include("abm.jl")  # Depends on: core, config, contacts
using .abm

const store = Dict{Symbol, Any}()  # Used in the loss function

function trainmodel(configfile::String)
    @info "$(now()) Configuring model"
    d   = YAML.load_file(configfile)
    cfg = Config(d)
    name2prior = construct_name2prior(d["unknowns"])

    @info "$(now()) Preparing training data"
    y = prepare_training_data(d)

    @info "$(now()) Initialising model"
    params = construct_params(cfg.paramsfile, cfg.demographics.params)
    model  = init_model(params, cfg)

    @info "$(now()) Populating convenience store"
    store[:y]          = y
    store[:ymax]       = maximum(y)
    store[:maxloss]    = rmse(y, fill(0, length(y)))
    store[:model]      = model
    store[:config]     = cfg
    store[:metrics]    = metrics
    store[:name2prior] = name2prior

    @info "$(now()) Training model"
    opts = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in d["options"])
    x, w = sparsegrid(size(name2prior, 1), opts[:order], gausslegendre)  # gausslegendre produces nodes in [-1, 1]
    rescale!(x, w, name2prior)  # Rescale nodes from the [-1, 1] cube to the box defined by the priors
    losses   = node_losses(x)
    gains    = 1.0 ./ losses
    probs    = gains ./ sum(gains)
    integral = dot(w, gains)

    @info "$(now()) Extracting result"
    @info "    Integral = $(integral)"
    @info "    Max loss = $(store[:maxloss])"
    colnames = [Symbol(colname) for (colname, prior) in name2prior]
    result   = construct_result(x, probs, losses, colnames)
    outfile  = joinpath(cfg.output_directory, "trained_params.tsv")
    CSV.write(outfile, result; delim='\t')
    @info "$(now()) Finished"
end

################################################################################
# Construct paramname => prior (orderd); Map a particle to params and policies

const registered_priors = Dict("uniform" => Uniform, "Normal" => Normal)

function construct_name2prior(arr)
    result = Pair{String, Distribution}[]  # name => prior
    for d in arr
       paramname, prior = first(d)
       priorname, args  = lowercase(prior[1]), prior[2:end]
       if haskey(registered_priors, priorname)
           priordist = registered_priors[priorname]
           push!(result, paramname => priordist(args...))
       else
           error("Unrecognised prior: $(prior)")
       end
    end
    result
end

function particle_to_params_and_policies!(particle, name2prior, cfg, modelparams)
    j = 0
    policies = (:distancingpolicy, :testingpolicy, :tracingpolicy, :quarantinepolicy)
    for (name, prior) in name2prior
        j += 1
        nameparts  = lowercase.(split(name, '.'))  # [category, name, date (optional)]
        category   = Symbol(nameparts[1])
        paramname  = Symbol(nameparts[2])
        paramvalue = particle[j]
        if category == :params
            modelparams[paramname] = paramvalue
        elseif in(category, policies)
            date      = Date(nameparts[3])
            fld       = Symbol("t2$(category)")
            dt2policy = getfield(cfg, fld)
            policy    = dt2policy[date]
            paramname = hasfield(typeof(policy), paramname) ? paramname : Symbol(uppercase(String(paramname)))
            setfield!(policy, paramname, paramvalue)
        else
            error("Unrecognised parameter category: $(category)")
        end
    end
end

################################################################################
# Scale nodes and weights from [-1, 1] cube to the box defined by the priors

function rescale!(x, w, name2prior)
    nparams, nnodes = size(x)
    w_scale = 1.0
    for i = 1:nparams
        prior     = name2prior[i][2]
        lb, ub    = params(prior)
        _scale    = 0.5 * (ub - lb)
        translate = 0.5 * (lb + ub)
        w_scale  *= _scale  # |Det(Jacobian)|
        for j = 1:nnodes
            x[i, j] = _scale*x[i, j] + translate
        end
    end
    w .*= w_scale
end

################################################################################
# Model run

function manyruns(model, cfg, metrics, ymax)
    firstday  = cfg.firstday
    lastday   = cfg.lastday
    agents    = model.agents
    daterange = firstday:Day(1):lastday
    result    = fill(0, length(daterange) - 1, cfg.nruns)  # Each column contains the time seris of 1 run
    for r in 1:cfg.nruns
        prev_totalpositives = 0  # Cumulative number of positives as of 11.59pm on date
        reset_model!(model, cfg)
        reset_metrics!(model)
        for (i, date) in enumerate(daterange)
            model.date = date
            date == lastday && break
            update_policies!(cfg, date)
            apply_forcing!(cfg.forcing, model, date)
            execute_events!(model.schedule[date], agents, model, date, metrics)

            # Collect result
            totalpositives = metrics[:positives]
            new_positives  = totalpositives - prev_totalpositives
            if new_positives > 2 * ymax  # EARLY STOPPING CRITERION: Too many cases
                return fill(Inf, 1, 1)   # Arbitrary 2D array containing large distances
            end
            result[i, r]        = new_positives
            prev_totalpositives = totalpositives
        end
    end
    result
end

################################################################################
# Loss function

"[loss(node) for node in nodes]"
function node_losses(nodes::Array{Float64, 2})
    nparams, nnodes = size(nodes)
    result = fill(store[:maxloss], nnodes)
    for j = 1:nnodes
        node = view(nodes, :, j)
        result[j] = loss(node)
        @info "$(now())    Node $(j). Loss = $(result[j])"
    end
    result
end

function loss(particle)
    particle_to_params_and_policies!(particle, store[:name2prior], store[:config], store[:model].params)
    yhat = manyruns(store[:model], store[:config], store[:metrics], store[:ymax])
    !isfinite(yhat[1, 1]) && return store[:maxloss]
    agg = [quantile(view(yhat, t, :), 0.5) for t = 1:size(yhat, 1)]  # Median of simulated values at each time step
    rmse(store[:y], agg)
end

function negative_loglikelihood(y, yhat)
    n = size(y, 1)
    result = 0.0
    for i = 1:n
        yhat[i] < 0.01 && continue
        result += y[i] * log(yhat[i]) - yhat[i]  # LL(Y ~ Poisson(yhat)) without constant term
    end
    -result / n
end

function rmse(y, yhat)
    n = size(y, 1)
    result = 0.0
    for i = 1:n
        result += abs2(y[i] - yhat[i])
    end
    sqrt(result / n)
end

################################################################################
# Utils

function construct_params(paramsfile::String, demographics_params::Dict{Symbol, Float64})
    tbl    = DataFrame(CSV.File(paramsfile))
    params = Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(tbl.name, tbl.value))
    merge!(params, demographics_params)  # Merge d2 into d1 and return d1 (d1 is enlarged, d2 remains unchanged)
end

"Returns: Vector{Int} containing the number of new cases"
function prepare_training_data(d)
    data      = DataFrame(CSV.File(d["training_data"]))  # Columns: date, newpositives, positives
    date2y    = Dict(date => y for (date, y) in zip(data.date, data.newpositives))
    firstday  = Date(d["firstday"])
    lastday   = Date(d["lastday"])
    daterange = firstday:Day(1):(lastday - Day(1))
    result    = fill(0, length(daterange))
    for (i, date) in enumerate(daterange)
        result[i] = haskey(date2y, date) ? date2y[date] : 0
    end
    result
end

function construct_result(nodes, probs, losses, colnames)
    nparams, nnodes = size(nodes)
    result = DataFrame(node=collect(1:nnodes), loss=fill(0.0, nnodes), prob=fill(0.0, nnodes))
    for colname in colnames
        result[!, colname] = fill(0.0, nnodes)
    end
    for j = 1:nnodes
        result[j, :loss] = losses[j]
        result[j, :prob] = probs[j]
        for (i, colname) in enumerate(colnames)
            result[j, colname] = nodes[i, j]  # ith component of jth node
        end
    end
    sort!(result, [:prob], rev=true)
end

end