module core

export Model,
       init_schedule, init_output, reset_output!,
       schedule!, execute_event!, execute_events!, metrics_to_output!,
       update_struct!

using DataFrames
using Dates

mutable struct Model{A}
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

function execute_events!(schedule, date, agents, model, metrics)
    !haskey(schedule, date) && return  # No events scheduled for date
    events = schedule[date]
    k = 1
    while haskey(events, k)
        func, id = events[k]
        execute_event!(func, agents[id], model, date, metrics)
        k += 1
    end
    empty!(events)
end

function init_output(metrics, firstday::Date, lastday::Date)
    naddresses = length(metrics)
    rg = firstday:Day(1):lastday
    n  = length(rg) * (naddresses + 1)  # +1 for sum over addresses
    result     = DataFrame(run=fill(0, n), date=Vector{Date}(undef, n), address=fill(0, n))
    address, m = first(metrics)
    for (colname, val0) in m
        result[!, colname] = Vector{typeof(val0)}(undef, n)
    end
    result
end

function metrics_to_output!(metrics, output, run_number::Int, dt::Date)
    # Init results for total over all addresses ("address" = 0)
    i1 = findfirst(isequal(0), output.run)
    output[i1, :run]  = run_number
    output[i1, :date] = dt

    # Results by address
    i = i1
    for (address, m) in metrics
        i += 1
        output[i, :run]     = run_number
        output[i, :date]    = dt
        output[i, :address] = address
        for (colname, val) in m
            output[i,  colname]  = val
            output[i1, colname] += val  # Populate total over all addresses
        end
    end
end

function reset_output!(output::DataFrame)
    for (colname, coldata) in eachcol(output, true)
        if colname == :date
            fill!(coldata, Date(1900, 1, 1))
        else
            fill!(coldata, zero(eltype(coldata)))
        end
    end
end

# To be extended by model-specific methods
unfit!(metrics, agent)  = nothing
fit!(metrics, agent)    = nothing

################################################################################
# Utils

function update_struct!(target, newvalue)
    flds = fieldnames(typeof(newvalue))
    for fld in flds
        setfield!(target, fld, getfield(newvalue, fld))
    end
end

end