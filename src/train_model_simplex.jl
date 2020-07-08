module train_model

export trainmodel

using CSV
using DataFrames
using Dates
using Distributions
using Logging
using YAML

include("abm.jl")  # Depends on: core, config, contacts
using .abm

include("simplex_integration.jl")
using .simplex_integration

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
    store[:neval]      = 0

    @info "$(now()) Training model"
    opts    = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in d["options"])
    gain(x) = exp(6.0 - loss(x))

    domain  = [(0.0, 0.05), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5), (0.0, 0.5)]
    npoints = 10
    x0      = [0.034, 0.5, 0.2, 0.2, 0.1, 0.1,  0.2]
    rtol    = 0.5
    res     = integrate(gain, domain, npoints, x0, rtol)

    @info "$(now()) Extracting result"
    @info "    Integral = $(res.integral)"
    losses   = 6.0 .- log.(res.heights)
    probs    = res.heights ./ sum(res.heights)
    colnames = [Symbol(colname) for (colname, prior) in name2prior]
    result   = construct_result(res.points, losses, probs, colnames)  # Columns: node, weight, loss, prob, params...
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

function loss(particle)
    store[:neval] += 1
    particle_to_params_and_policies!(particle, store[:name2prior], store[:config], store[:model].params)
    yhat = manyruns(store[:model], store[:config], store[:metrics], store[:ymax])
    if isfinite(yhat[1, 1])
        agg    = [quantile(view(yhat, t, :), 0.5) for t = 1:size(yhat, 1)]  # Median of simulated values at each time step
        result = rmse(store[:y], agg)
        result = result < store[:maxloss] ? result : 1000.0  # gain = exp(-loss) = exp(-1000) = 0
    else
        result = 1000.0
    end
    @info "$(now())    Eval $(store[:neval]). Loss = $(round(result; digits=4)). x = $(round.(particle; digits=4))"
    result
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

function construct_result(points, losses, probs, colnames)
    nparams, npoints = size(points)
    result = DataFrame(node=collect(1:npoints), loss=losses, prob=probs)
    for (i, colname) in enumerate(colnames)
        result[!, colname] = [x for x in view(points, i, :)]
    end
    sort!(result, [:prob], rev=true)
end

end