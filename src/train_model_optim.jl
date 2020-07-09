module train_model

export trainmodel

using CSV
using DataFrames
using Dates
using Distributions
using Logging
using Optim
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
    store[:trace]      = NamedTuple{(:loss, :x),Tuple{Float64,Vector{Float64}}}[]

    @info "$(now()) Training model"
    nparams = size(name2prior, 1)
    theta0  = fill(0.0, nparams)
theta0 = [0.034, 0.5, 0.2, 0.2, 0.1, 0.1, 0.2]
    opts    = Optim.Options(; Dict{Symbol, Any}(Symbol(k) => v for (k, v) in d["options"])...)
    result  = optimize(loss, theta0, NelderMead(), opts)

    @info "$(now()) Extracting result"
    colnames = [Symbol(colname) for (colname, prior) in name2prior]
    result   = construct_result(store[:trace], colnames)  # Columns: node, weight, loss, prob, params...
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
    particle_to_params_and_policies!(particle, store[:name2prior], store[:config], store[:model].params)
    yhat = manyruns(store[:model], store[:config], store[:metrics], store[:ymax])
    if isfinite(yhat[1, 1])
        agg    = [quantile(view(yhat, t, :), 0.5) for t = 1:size(yhat, 1)]  # Median of simulated values at each time step
        result = rmse(store[:y], agg)
        result = result < store[:maxloss] ? result : 1000.0  # gain = exp(-loss) = exp(-1000) = 0
    else
        result = 1000.0
    end
    push!(store[:trace], (loss=result, x=particle))
    @info "$(now())    Eval $(length(store[:trace])). Loss = $(round(result; digits=4)). x = $(round.(particle; digits=4))"
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

function construct_result(trace, colnames)
    # First pass: All rows of trace
    losses = [point.loss for point in trace]
    temp   = DataFrame(loss=losses)
    for (i, colname) in enumerate(colnames)
        temp[!, colname] = [point.x[i] for point in trace]
    end

    # Second pass: Group duplicate particles
    result = DataFrame(fill(Float64, length(colnames) + 1), vcat(:loss, colnames), 0)
    for subdata in groupby(temp, colnames)
        row = Dict{Symbol, Float64}(:loss => mean(subdata.loss))
        for colname in colnames
           row[colname] = subdata[1, colname]
        end
        push!(result, row)
    end
    minloss = minimum(result.loss)
    gains   = exp.(minloss .- result.loss)
    result[!, :node] = collect(1:size(result,1))
    result[!, :prob] = gains ./ sum(gains)
    result = result[:, vcat(:node, :loss, :prob, colnames)]
    sort!(result, [:prob], rev=true)
end

end