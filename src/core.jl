module core

export Model, AbstractAgent, AbstractEventBag,  # types
       run!, schedule!

using DataFrames
using Distributions

abstract type AbstractAgent end
abstract type AbstractEventBag end  # An instance is a set of events that occur simultaneously

mutable struct Model{A <: AbstractAgent, E <: AbstractEventBag, T <: NamedTuple}
    agents::Vector{A}
    params::T
    time::Int
    maxtime::Int
    schedule0::E         # Events that take place during (0, 1)
    schedule::Vector{E}  # Events that take place during (1, maxtime)
end

"Schedule an event at time t for agent id"
function schedule!(agent_id, t, event::Symbol, model)
    eb = t == 0 ? model.schedule0 : model.schedule[t]
    func, ids = getfield(eb, event)
    push!(ids, agent_id)
end

"Execute an event bag"
function execute!(eb::T, model, t::Int) where {T <: AbstractEventBag}
    agents = model.agents
    for fname in fieldnames(typeof(eb))
        func, ids = getfield(eb, fname)
        for id in ids
            func(agents[id], model, t)
        end
    end
end

function run!(model)
    output, status2rownum = init_output(model)
    collectdata!(model, output, 0, status2rownum)  # t == 0
    execute!(model.schedule0, model, 0)  # t in (0, 1)
    for t = 1:(model.maxtime - 1)
        model.time = t
        collectdata!(model, output, t, status2rownum)
        execute!(model.schedule[t], model, t)
    end
    collectdata!(model, output, model.maxtime, status2rownum)
    output_to_dataframe(output, status2rownum)
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

function output_to_dataframe(output, status2rownum)
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
    result
end

end