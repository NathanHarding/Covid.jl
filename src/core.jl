module core

export Model, AbstractAgent,  # Types
       init_schedule, init_output, reset_output!,
       schedule!, execute_event!, execute_events!, metrics_to_output!,
       update_struct!  # Utils

using DataFrames
using Dates

abstract type AbstractAgent end

mutable struct Model{A <: AbstractAgent}
    agents::Vector{A}
    params::Dict{Symbol, Float64}
    date::Date
    lastday::Date
    schedule::Dict{Date, Dict{Int, Tuple{Function, Int}}}  # t => i => event, where i denotes the order of events
end

init_schedule(firstday::Date, lastday::Date) = Dict(dt => Dict{Int, Tuple{Function, Int}}() for dt = firstday:Day(1):(lastday - Day(1)))

"Schedule an event at time t for agent id"
function schedule!(agentid::Int, dt::Date, event::Function, model)
    dt >= model.lastday && return
    s = model.schedule[dt]
    n = length(s)
    s[n+1] = (event, agentid)
end

function execute_event!(func::Function, agent, model, dt, metrics)
    unfit!(metrics, agent)  # Remove agent's old state from metrics
    func(agent, model, dt)
    fit!(metrics, agent)    # Add agent's new state to metrics
end

function execute_events!(events, agents, model, dt, metrics)
    k = 1
    while haskey(events, k)
        func, id = events[k]
        execute_event!(func, agents[id], model, dt, metrics)
        k += 1
    end
    empty!(events)
end

function init_output(metrics, firstday::Date, lastday::Date)
    rg = firstday:Day(1):lastday
    n  = length(rg)
    result = DataFrame(run=fill(0, n), date=[dt for dt in rg])
    for (colname, val0) in metrics
        result[!, colname] = Vector{typeof(val0)}(undef, n)
    end
    result
end

function reset_output!(output::DataFrame, run_number::Int)
    fill!(output.run, run_number)
    for (colname, coldata) in eachcol(output, true)
        colname == :run  && continue
        colname == :date && continue
        fill!(coldata, zero(eltype(coldata)))
    end
end

function metrics_to_output!(metrics, output, dt::Date)
    i = findfirst(isequal(dt), output.date)
    for (colname, val) in metrics
        output[i, colname] = val
    end
end

# To be extended by model-specific methods
unfit!(metrics, agents)  = nothing
fit!(metrics, agents)    = nothing

################################################################################
# Utils

function update_struct!(target, newvalue)
    flds = fieldnames(typeof(newvalue))
    for fld in flds
        setfield!(target, fld, getfield(newvalue, fld))
    end
end

end