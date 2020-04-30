module core

export Model, AbstractAgent,  # types
       run!, schedule!

using DataFrames
using Distributions

abstract type AbstractAgent end

mutable struct Model{A <: AbstractAgent, T <: NamedTuple}
    agents::Vector{A}
    params::T
    time::Int
    maxtime::Int
    schedule0::Vector{Tuple{Function, Int}}         # t=0: schedule0   = [(function_01, agentid_01), ...]
    schedule::Vector{Vector{Tuple{Function, Int}}}  # t>0: schedule[t] = [(function_t1, agentid_t1), ...]
end

"Schedule an event at time t for agent id"
function schedule!(agentid::Int, t, event::Function, model)
    if t == 0
        push!(model.schedule0, (event, agentid))
    else
        push!(model.schedule[t], (event, agentid))
    end
end

"Execute a Vector of events"
function execute!(events::Vector{Tuple{Function, Int}}, model, t::Int, scenario)
    agents = model.agents
    for (func, id) in events
        agent = agents[id]
        #unfit!(metrics, agent)  # Remove agent's old state from metrics
        func(agent, model, t, scenario)
        #fit!(metrics, agent)    # Add agent's new state to metrics
    end
end

function run!(model, scenario, run_number::Int)
    output, status2rownum = init_output(model)
    collectdata!(model, output, 0, status2rownum)  # t == 0
    execute!(model.schedule0, model, 0, scenario)  # t in (0, 1)
    for t = 1:(model.maxtime - 1)
        model.time = t
        collectdata!(model, output, t, status2rownum)
        execute!(model.schedule[t], model, t, scenario)
    end
    collectdata!(model, output, model.maxtime, status2rownum)
    output_to_dataframe(output, status2rownum, run_number)
end

################################################################################
# Output

function init_output(model)
    output = fill(0.0, 8, model.maxtime + 1)  # output[:, t] is the observation (SEIHCVRD) at time t
    status2rownum = Dict(:S => 1, :E => 2, :I => 3, :H => 4, :C => 5, :V => 6, :R => 7, :D => 8)
    output, status2rownum
end

function collectdata!(model, output, t, status2rownum)
    j = t + 1
    agents = model.agents
    for agent in agents
        output[status2rownum[agent.status], j] += 1
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