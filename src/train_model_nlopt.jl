module train_model

export trainmodel

using CSV
using DataFrames
using Dates
using Distributions
using Logging
using NLopt
using YAML

include("abm.jl")  # Depends on: core, config, contacts
using .abm

const store = Dict{Symbol, Any}()  # Used in the loss function

function trainmodel(configfile::String)
    @info "$(now()) Configuring model"
    d   = YAML.load_file(configfile)
    cfg = Config(d)
    daterange  = cfg.firstday:Day(1):cfg.lastday
    @info "Training based on $(daterange)"
    name2prior = construct_name2prior(d["unknowns"])
    nparams    = size(name2prior, 1)

    @info "$(now()) Preparing training data"
    y = prepare_training_data(d)

    @info "$(now()) Initialising model"
    params = construct_params(cfg.paramsfile)
    model  = init_model(params, cfg)

    @info "$(now()) Initialising metrics"
    initialise_metrics(model.agents)

    @info "$(now()) Populating convenience store"
    store[:totalpositives] = sum(y.newpositives)  # Final cumulative number of detected cases
    store[:totaldeaths]    = sum(y.newdeaths)     # Final cumulative number of deaths
    store[:y]          = y
    store[:yhat]       = init_yhat(daterange)
    store[:stats]      = init_stats(length(daterange), 2, cfg.nruns)
    store[:maxloss]    = loss(store[:y], store[:yhat])
    store[:model]      = model
    store[:config]     = cfg
    store[:metrics]    = metrics
    store[:name2prior] = name2prior
    store[:trace]      = NamedTuple{(:loss, :x), Tuple{Float64,Vector{Float64}}}[]
    store[:output]     = init_output(metrics, cfg.firstday, cfg.lastday)
    store[:daterange]  = daterange
    store[:date2i]     = Dict(date => i for (i, date) in enumerate(daterange))

    @info "$(now()) Training model (maxloss = $(round(store[:maxloss]; digits=4)))"
    # opt = Opt(:GN_DIRECT_L, nparams)
    opt = Opt(:GN_CRS2_LM, nparams)
    #opt = Opt(:G_MLSL_LDS, nparams)
    #opt.local_optimizer = Opt(:LN_SBPLX, nparams)
    opt.min_objective = lossfunc
    lb, ub            = getbounds(name2prior)
    opt.lower_bounds  = lb
    opt.upper_bounds  = ub
    if !isnothing(d["options"])
        for (k, v) in d["options"]
            setproperty!(opt, Symbol(k), v)
        end
    end
    theta0 = [mean(prior) for (name, prior) in name2prior]
    (fmin, xmin, ret) = optimize(opt, theta0)
    @info "$(now()) Return code = $(ret). fmin = $(fmin).\n  xmin = $(xmin)"

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
        elseif category == :meta
            date      = Date(nameparts[3])
            if paramname == :no_home
                fld = Symbol("t2distancingpolicy")
                dt2policy = getfield(cfg, fld)
                policy    = dt2policy[date]
                paramname = Symbol(:school)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:workplace)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:community)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:social)
                setfield!(policy, paramname, paramvalue)
            elseif paramname == :no_home_fr
                fld = Symbol("t2distancingpolicy")
                dt2policy = getfield(cfg, fld)
                policy    = dt2policy[date]
                paramname = Symbol(:school)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:workplace)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:community)
                setfield!(policy, paramname, 0.5 * paramvalue)
                paramname = Symbol(:social)
                setfield!(policy, paramname, 0.5 * paramvalue)
            elseif paramname == :struc
                fld = Symbol("t2distancingpolicy")
                dt2policy = getfield(cfg, fld)
                policy    = dt2policy[date]
                paramname = Symbol(:school)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:workplace)
                setfield!(policy, paramname, paramvalue)
            elseif paramname == :mix
                fld = Symbol("t2distancingpolicy")
                dt2policy = getfield(cfg, fld)
                policy    = dt2policy[date]
                paramname = Symbol(:community)
                setfield!(policy, paramname, paramvalue)
                paramname = Symbol(:social)
                setfield!(policy, paramname, paramvalue)
            elseif paramname == :home_const
                fld = Symbol("t2distancingpolicy")
                dt2policy = getfield(cfg, fld)
                dates = [Date(2020,06,01),Date(2020,07,08),Date(2020,08,02)]
                for dat in dates
                    policy    = dt2policy[dat]
                    paramname = Symbol(:household)
                    setfield!(policy, paramname, paramvalue)
                end
            end
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

function manyruns(model, cfg, metrics, output, stats, totalpositives, date2i)
    firstday  = cfg.firstday
    daterange = firstday:Day(1):cfg.lastday
    agents    = model.agents
    for r in 1:cfg.nruns
        reset_model!(model, cfg)
        reset_metrics!(model)
        reset_output!(output)
        i_output = 0  # The number of rows of output that have been populated
        for (i, date) in enumerate(daterange)
            model.date = date
            update_policies!(cfg, date, date > firstday)
            apply_forcing!(cfg.forcing, model, date, cfg.cumsum_population)
            execute_events!(model.schedule, date, agents, model, metrics)
            i_output, i_total = metrics_to_output!(metrics, output, r, date, i_output)  # System at 11:59pm
            output[i_total, :positives] >= 2*totalpositives && return false  # EARLY STOPPING CRITERION: Too many cases
            output_to_stats!(stats, output, i_total, r, date2i)
        end
    end
    true
end

################################################################################
# Loss function

"NLopt requires the gradient as the 2nd argument, even though we don't use it."
function lossfunc(particle, grad)
    particle_to_params_and_policies!(particle, store[:name2prior], store[:config], store[:model].params)
    maxloss = 100000.0  # gain = exp(-loss) = exp(-1000) = 0
    ok      = manyruns(store[:model], store[:config], store[:metrics], store[:output], store[:stats], store[:totalpositives], store[:date2i])
    if ok
        stats_to_yhat!(store[:yhat], store[:stats])
        result = loss(store[:y], store[:yhat])
        result = result >= store[:maxloss] ? maxloss : result
    else
        result = maxloss
    end
    push!(store[:trace], deepcopy((loss=result, x=particle)))
    @info "$(now())    Eval $(length(store[:trace])). Loss = $(round(result; digits=4)). x = $(round.(particle; digits=4))"
    result
end

function loss(y::DataFrame, yhat::DataFrame)
    1.0 * rmse(y.newpositives, yhat.newpositives) +
    0.0 * rmse(y.newdeaths, yhat.newdeaths)
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

function mae(y, yhat)
    n = size(y, 1)
    result = 0.0
    for i = 1:n
        result += abs(y[i] - yhat[i])
    end
    result / n
end

function rmse_roll(y, yhat)
    n = size(y, 1)
    result = 0.0
    for i = 1:n-2
        result += abs2(y[i]+y[i+1]+y[i+2] - yhat[i]-yhat[i+1]-yhat[i+2])
    end
    sqrt(result / n)
end

################################################################################
# metrics -> output (DataFrame) -> stats (Array{Int, 3}) -> yhat (DataFrame)
# yhat (y and yhat are the inputs to the loss function)

"Returns: DataFrame with columns = date, newpositives, newdeaths. To be populated with median (over runs) values."
function init_yhat(daterange)
    DataFrame(date=collect(daterange), newpositives=fill(0.0, length(daterange)), newdeaths=fill(0.0, length(daterange)))
end

"ndates x nvars x nruns"
init_stats(ndates, nvars, nruns) = fill(0, ndates, nvars, nruns)

function output_to_stats!(stats, output, i_total, run_number, date2i)
    date = output[i_total, :date]
    i    = date2i[date]
    stats[i, 1, run_number] = output[i_total, :positives]
    stats[i, 2, run_number] = output[i_total, :D]
end

"Populate yhat with the values collected in stats."
function stats_to_yhat!(yhat::DataFrame, stats)
    diff!(stats)  # Convert cumulative totals into daily incidence
    for (i, date) in enumerate(yhat.date)
        yhat[i, :newpositives] = quantile(view(stats, i, 1, :), 0.5)
        yhat[i, :newdeaths]    = quantile(view(stats, i, 2, :), 0.5)
    end
end

function diff!(stats::Array{Int, 3})
    ni, nvars, nruns = size(stats)
    for r = 1:nruns
        for j = 1:nvars
            diff!(view(stats, :, j, r))
        end
    end
end

function diff!(v::T) where {T <: AbstractVector}
    n = size(v, 1)
    for i = 2:n
        v[i] -= v[i - 1]
    end
end

################################################################################
# Utils

function construct_params(paramsfile::String)
    tbl = DataFrame(CSV.File(paramsfile))
    Dict{Symbol, Float64}(Symbol(k) => v for (k, v) in zip(tbl.name, tbl.value))
end

function getbounds(name2prior)
    nparams = size(name2prior, 1)
    lb = fill(0.0, nparams)
    ub = fill(0.0, nparams)
    i  = 0
    for (name, prior) in name2prior
        i    += 1
        a, b  = Distributions.params(prior)
        lb[i] = a
        ub[i] = b
    end
    lb, ub
end

"Returns: DataFrame with columns: date, newpositives, newdeaths."
function prepare_training_data(d)
    # Import training data
    opts      = d["training_data"]
    data      = DataFrame(CSV.File(opts["filename"]))
    datecol   = Symbol(opts["datecolumn"])
    datefmt   = opts["dateformat"]
    deathdate = Symbol(opts["deathdate"])
    data[!, datecol]   = [Date(x, datefmt) for x in data[!, datecol]]
    data[!, deathdate] = [ismissing(x) ? missing : Date(x, datefmt) for x in data[!, deathdate]]
    sort!(data, [datecol])
    
    # Construct result
    firstday  = Date(d["firstday"])
    lastday   = Date(d["lastday"])
    daterange = firstday:Day(1):lastday
    n         = length(daterange)
    result    = DataFrame(date=collect(daterange), newpositives=fill(0, n), newdeaths=fill(0, n))
    for (i, date) in enumerate(daterange)
        v1 = view(data, (.!ismissing.(data[!, datecol]))   .& (data[!, datecol]   .== date), :)
        v2 = view(data, (.!ismissing.(data[!, deathdate])) .& (data[!, deathdate] .== date), :)
        result[i, :newpositives] = size(v1, 1)
        result[i, :newdeaths]    = size(v2, 1)
    end
    result
    show(result)
end

function construct_result(trace, colnames)
    # First pass: All rows of trace
    losses = [point.loss for point in trace]
    temp   = DataFrame(loss=losses)
    for (i, colname) in enumerate(colnames)
        temp[!, colname] = [point.x[i] for point in trace]
    end

    # Second pass: Group duplicate particles
    result = DataFrame(fill(Float64, 1 + length(colnames)), vcat(:loss, colnames), 0)
    a = groupby(temp,colnames)
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