module core

export Model, AbstractAgent,  # types
       run!, init_schedule, schedule!

using DataFrames
using Distributions

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
function execute!(events, model, t::Int, scenario, metrics)
    agents = model.agents
    n = length(events)
    for i = 1:n
        func, id = events[i]
        agent = agents[id]
        unfit!(metrics, agent)  # Remove agent's old state from metrics
        func(agent, model, t, scenario)
        fit!(metrics, agent)    # Add agent's new state to metrics
    end
end

function run!(model, scenario, run_number::Int)
    metrics = init_metrics(model)
    output, status2rownum = init_output(model)
    for t = 0:(model.maxtime - 1)
        model.time = t
        metrics_to_output!(metrics, output, t, status2rownum)     # System as at t
        execute!(model.schedule[t], model, t, scenario, metrics)  # Events that occur in (t, t+1)
    end
    metrics_to_output!(metrics, output, model.maxtime, status2rownum)
    output_to_dataframe(output, status2rownum, run_number)
end

################################################################################
# Output

function init_metrics(model)
    metrics = Dict(:S => 0, :E => 0, :I => 0, :H => 0, :C => 0, :V => 0, :R => 0, :D => 0)
    for agent in model.agents
        metrics[agent.status] += 1
    end
    metrics
end

function unfit!(metrics, agent)
    metrics[agent.status] -= 1
end

function fit!(metrics, agent)
    metrics[agent.status] += 1
end

function init_output(model)
    output = fill(0.0, 8, model.maxtime + 1)  # output[:, t] is the observation (SEIHCVRD) at time t
    status2rownum = Dict(:S => 1, :E => 2, :I => 3, :H => 4, :C => 5, :V => 6, :R => 7, :D => 8)
    output, status2rownum
end

function metrics_to_output!(metrics, output, t, status2rownum)
    j = t + 1
    for (status, n) in metrics
        output[status2rownum[status], j] = n
    end
end

function output_to_dataframe(output, status2rownum, run_number::Int)
    ni = size(output, 2)
    nj = size(output, 1)
    j2name   = Dict(v => k for (k, v) in status2rownum)
    colnames = vcat(:time, [j2name[j] for j = 1:nj])
    result   = DataFrame(fill(Int, length(colnames)), colnames, ni)
    for i = 1:ni
        result[i, :time] = i - 1
        for (j, colname) in j2name
            result[i, colname] = output[j, i]
        end
    end
    result[!, :run] = fill(run_number, size(result, 1))
    result
end

end