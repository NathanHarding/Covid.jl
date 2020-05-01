module core

export Model, AbstractAgent,  # types
       init_schedule, init_output, reset_output!,
       schedule!, execute_event!, execute_events!, metrics_to_output!

using DataFrames

abstract type AbstractAgent end

mutable struct Model{A <: AbstractAgent, T <: NamedTuple}
    agents::Vector{A}
    params::T
    time::Int
    maxtime::Int
    schedule::Dict{Int, Dict{Int, Tuple{Function, Int}}}  # t => i => event, where i denotes the order of events
end

init_schedule(maxtime::Int) = Dict(t => Dict{Int, Tuple{Function, Int}}() for t = 0:(maxtime-1))

"Schedule an event at time t for agent id"
function schedule!(agentid::Int, t, event::Function, model)
    s = model.schedule[t]
    n = length(s)
    s[n+1] = (event, agentid)
end

function execute_event!(func::Function, agent, model, t, metrics)
    unfit!(metrics, agent)  # Remove agent's old state from metrics
    func(agent, model, t)
    fit!(metrics, agent)    # Add agent's new state to metrics
end

function execute_events!(events, agents, model, t, metrics)
    k = 1
    while haskey(events, k)
        func, id = events[k]
        execute_event!(func, agents[id], model, t, metrics)
        k += 1
    end
end

unfit!(metrics, agents) = nothing  # To be extended by model-specific method
fit!(metrics, agents)   = nothing  # To be extended by model-specific method

function init_output(metrics, n)
    result = DataFrame(run=fill(0, n), time=[i for i = 0:(n-1)])
    for (colname, val0) in metrics
        result[!, colname] = Vector{typeof(val0)}(undef, n)
    end
    result
end

function reset_output!(output::DataFrame, run_number::Int)
    fill!(output.run, run_number)
    for (colname, coldata) in eachcol(output, true)
        colname == :run  && continue
        colname == :time && continue
        fill!(coldata, zero(eltype(coldata)))
    end
end

function metrics_to_output!(metrics, output, t)
    i = t + 1
    for (colname, val) in metrics
        output[i, colname] = val
    end
end

end