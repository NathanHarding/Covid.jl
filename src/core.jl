module core

export Model, AbstractAgent,  # types
       init_schedule, init_output, reset_output!,
       schedule!, run!

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

"Execute a Vector of events"
function execute!(events, model, t::Int, scenario, metrics, f_unfit::Function, f_fit::Function)
    agents = model.agents
    n = length(events)
    for i = 1:n
        func, id = events[i]
        agent = agents[id]
        f_unfit(metrics, agent)  # Remove agent's old state from metrics
        func(agent, model, t, scenario)
        f_fit(metrics, agent)    # Add agent's new state to metrics
    end
end

function run!(model, scenario, run_number::Int, metrics, output, f_unfit::Function, f_fit::Function)
    maxtime  = model.maxtime
    schedule = model.schedule
    for t = 0:(maxtime - 1)
        model.time = t
        metrics_to_output!(metrics, output, t)  # System as at t
        execute!(schedule[t], model, t, scenario, metrics, f_unfit, f_fit)  # Events that occur in (t, t+1)
    end
    metrics_to_output!(metrics, output, maxtime)
end

function init_output(metrics, n)
    result = DataFrame()
    for (colname, val0) in metrics
        result[!, colname] = Vector{typeof(val0)}(undef, n)
    end
    result
end

function reset_output!(output::DataFrame)
    for col in eachcol(output)
        fill!(col, zero(eltype(col)))
    end
end

function metrics_to_output!(metrics, output, t)
    i = t + 1
    for (colname, val) in metrics
        output[i, colname] = val
    end
end

end