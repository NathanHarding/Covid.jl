using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Dates

"Filter large file, retaining rows with colname == value."
function filter_result(infile::String, outfile::String, colname::Symbol, value::String)
    result = Dict{Symbol, String}[]  # Each element is a row of the result. All types are strings...saves parsing
    for row in CSV.Rows(infile; delim=',', use_mmap=true, type=String)
        row[colname] != value && continue  # Row is excluded
        d = Dict{Symbol, String}()
        for cname in propertynames(row)
            d[cname] = row[cname]
        end
        push!(result, d)
    end
    result = DataFrame(result)
    CSV.write(outfile, result; delim=',')
end

infile  = "data\\output\\metrics.csv"
outfile = "data\\output\\metrics_filtered.csv"
filter_result(infile, outfile, :address, "0")