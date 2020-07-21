using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Dates
using Distributions

"Return the desired quantile (over the runs) of the results on each date."
function result_quantile(infile::String, outfile::String, q::Float64)
    data     = DataFrame(CSV.File(infile))
    colnames = Set(Symbol.(names(data)))
    pop!(colnames, :run)
    pop!(colnames, :date)
    pop!(colnames, :address)
    coltypes = vcat(eltype(data.address), Date, fill(Float64, length(colnames)))  # Floats allow for non-integer quantiles
    allcols  = vcat([:address, :date], [x for x in colnames])
    result   = DataFrame(coltypes, allcols, 0)
    for subdata in groupby(data, [:address, :date])
        address = subdata[1, :address]
        date    = subdata[1, :date]
        row     = Dict{Symbol, Any}(:address => address, :date => date)
        for colname in colnames
            row[colname] = quantile(subdata[!, colname], q)
        end
        push!(result, row)
    end
    CSV.write(outfile, result; delim=',')
end

infile  = "data\\output\\metrics_no_intervention.csv"
outfile = "data\\output\\median_no_intervention.csv"
result_quantile(infile, outfile, 0.5)