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

const ystats = Dict{Symbol, Int}(:max => 0, :total => 0)  # Used for early stopping criterion

function trainmodel(configfile::String)
    @info "$(now()) Configuring model"
    d   = YAML.load_file(configfile)
    cfg = Config(d)
    unknowns, n_unknowns = construct_unknowns(d["unknowns"])

    @info "$(now()) Preparing training data"
    y = prepare_training_data(d)
    ystats[:max]   = maximum(y)
    ystats[:total] = sum(y)

    @info "$(now()) Initialising model"
    params = construct_params(cfg.paramsfile, cfg.demographics.params)
    model  = init_model(params, cfg)

    @info "$(now()) Training model"
    opts = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in d["solver_options"])
    probs, weights, distances = beaumont2009(y, model, cfg, metrics, unknowns, n_unknowns, opts)

    @info "$(now()) Extracting result"
    result  = construct_result(probs, weights, distances)
    outfile = joinpath(cfg.output_directory, "trained_params.tsv")
    CSV.write(outfile, result; delim='\t')
    @info "$(now()) Finished"
end

################################################################################
# ABC

logit_to_prob(b, pmax) = pmax / (1.0 + exp(-b))  # 0 <= result <= pmax
#prob_to_logit(p) = log(p / (1.0 - p))

"""
Probabilities are calculated from logits, which are drawn from independent Normals. 
"""
function beaumont2009(y, model, cfg, metrics, unknowns, n_unknowns, opts)
    # Extract options
    nparams    = n_unknowns
    nparticles = opts[:nparticles]
    nsteps     = opts[:nsteps]
    pmax       = opts[:pmax]
    alpha      = opts[:alpha]  # Quantile of distances used to set the distance threshold in steps 2:nsteps

    # Work spaces
    probs     = fill(0.0, nparams, nparticles, nsteps)  # probs[:, j] = logit_to_prob.(particles[:, j, t], pmax)
    particles = fill(0.0, nparams, nparticles, nsteps)  # particles[:, j, t] ~ MvNormal(means[j], stds). Logit scale.  
    means     = fill(0.0, nparams, nparticles)          # Mean of particle proposal distribution
    stds      = fill(1.0, nparams)
    weights   = fill(0.0, nparticles, nsteps)
    distances = fill(Inf, nparticles, nsteps)

    @info "$(now()) T = 1. epsilon = Inf"
    for j = 1:nparticles
        p = view(probs, :, j, 1)
        for r = 1:1000  # Obtain a particle with finite loss
            sample_particle!(p, view(particles, :, j, 1), view(means, :, j), stds, pmax)
            output = manyruns(p, model, cfg, metrics, unknowns)
            distances[j, 1] = loss(y, output)
            if isfinite(distances[j, 1])
                println("    Particle $(j). Distance = $(distances[j, 1])")
                update_mean!(means, particles, j, 1)  # Set means[:, j] = particles[:, j, t]
                weights[j, 1] = 1.0 / nparticles
                break
            end
        end
    end
    update_stds!(stds, particles, weights, 1)

    # t > 1
    for t = 2:nsteps
        epsilon = quantile(view(distances, :, t-1), alpha)
        @info "$(now()) T = $(t). epsilon = $(epsilon)"
        w_dist  = Categorical(weights[:, t-1])
        for j = 1:nparticles
            p = view(probs, :, j, t)
            for r = 1:1000  # Obtain a particle with finite loss
                k = rand(w_dist)  # Select a particle at random
                distances[k, t-1] > epsilon && continue  # Only sample particles with distance <= epsilon
                sample_particle!(p, view(particles, :, j, t), view(means, :, k), stds, pmax)  # Set jth particle = Perturbed kth particle
                output = manyruns(p, model, cfg, metrics, unknowns)
                dist   = loss(y, output)
                if dist <= epsilon
                    println("    Particle $(j). Distance = $(dist)")
                    distances[j, t] = dist
                    update_mean!(means, particles, j, t)  # Set means[:, j] = particles[:, j, t]
                    update_weight!(weights, j, t, particles, stds)
                    break
                end
            end
        end
        normalise!(view(weights, :, t))
        update_stds!(stds, particles, weights, t)
    end
    probs, weights, distances
end

"""
Sample from Normal(m, s) and store in particle.
Convert to probabilities and store in probs.
"""
function sample_particle!(probs, particle, m, s, pmax)
    nparams = length(m)
    for j = 1:nparams
        r = rand(Normal(m[j], s[j]))
        particle[j] = r
        probs[j]    = logit_to_prob(r, pmax)
    end
end

"Set means[:, j] = particles[:, j, t]"
function update_mean!(means, particles, j, t)
    nparams = size(means, 1)
    for i = 1:nparams
        means[i, j] = particles[i, j, t]
    end
end

function update_stds!(stds, particles, weights, t)
    nparams, nparticles, nsteps = size(particles)
    w = Distributions.StatsBase.AnalyticWeights(view(weights, :, t))
    for i = 1:nparams
        theta   = view(particles, i, :, t)  # Values of ith parameter at step t
        m       = mean(theta, w)
        stds[i] = sqrt(2.0 * var(theta, w; mean=m, corrected=true))  # Variance = 2 x empirical variance of ith parameter
    end
end

"Updates weights[j, t] but does not normalise weights[:, t]."
function update_weight!(weights, j, t, particles, stds)
    nparams    = size(particles, 1)
    nparticles = size(particles, 2)
    stdnormal  = Normal(0.0, 1.0)
    num = 1.0
    for i = 1:nparams
        num *= pdf(stdnormal, particles[i, j, t])  # prior(jth particle)
    end
    denom = 0.0
    for k = 1:nparticles
        w    = weights[k, t-1]
        dens = 1.0
        for i = 1:nparams
            z     = (particles[i, j, t] - particles[i, k, t-1]) / stds[i]
            dens *= pdf(stdnormal, z)
        end
        denom += w*dens
    end
    weights[j, t] = num / denom
end

function normalise!(v)
    v ./= sum(v)
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
        theta[i] = params[nm]
    end
    i
end

function theta_to_params!(theta, i, unknowns::Vector{Symbol}, params::Dict{Symbol, Float64}, )
    for nm in unknowns
        i += 1
        params[nm] = theta[i]
    end
    i
end

function policy_to_theta!(theta, i, unknowns::Vector{Pair{Date, Vector{Symbol}}}, dt2policy)
    for (dt, flds) in unknowns
        policy = dt2policy[dt]
        for fld in flds
            i += 1
            theta[i] = getfield(policy, fld)
        end
    end
    i
end

function theta_to_policy!(theta, i, unknowns::Vector{Pair{Date, Vector{Symbol}}}, dt2policy)
    for (dt, flds) in unknowns
        policy = dt2policy[dt]
        for fld in flds
            i += 1
            setfield!(policy, fld, theta[i])
        end
    end
    i
end

################################################################################
# Model run

function manyruns(theta, model, cfg, metrics, unknowns)
    theta_to_params_and_policies!(theta, unknowns, cfg, model.params)
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
            if new_positives > 2 * ystats[:max]  # EARLY STOPPING CRITERION: Too many cases
                return fill(Inf, 1, 1)  # Arbitrary 2D array containing large distances
            end
            result[i, r]        = new_positives
            prev_totalpositives = totalpositives
        end
    end
    result
end

################################################################################
# Loss function

function loss(y, yhat)
    !isfinite(yhat[1, 1]) && return Inf
    agg = [quantile(view(yhat, t, :), 0.5) for t = 1:size(yhat, 1)]  # Median of simulated values at each time step
    rmse(y, agg)
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

function construct_result(probs, weights, distances)
    # Init result
    nparams, nparticles, nsteps = size(probs)
    result = DataFrame(step=Int[], particle=Int[], weight=Float64[], distance=Float64[])
    for i = 1:nparams
        colname = Symbol("x$(i)")  # FIX
        result[!, colname] = Float64[]
    end

    # Populate result
    for t = 1:nsteps
        for j = 1:nparticles
            row = Dict{Symbol, Any}(:step => t, :particle => j, :weight => weights[j, t], :distance => distances[j, t])
            for i = 1:nparams
                colname = Symbol("x$(i)")
                row[colname] = probs[i, j, t]
            end
            push!(result, row)
        end
    end
    result
end

end