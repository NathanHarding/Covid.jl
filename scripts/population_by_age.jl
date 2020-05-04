#=
  Contents: Script for constructing population by age by SA2.
  The intermediate data is copied from the raw Excel spreadsheet sourced from the ABS.
=#

cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using CSV
using DataFrames

################################################################################
# Functions

function age_distribution(data::DataFrame)
    result   = DataFrame(age=Int[], count=Int[])
    colidx1  = findfirst(==(:age_0), names(data))
    colidx2  = size(data, 2)
    colnames = names(data)
    ntotal   = 0
    for j = colidx1:colidx2
        colname = string(colnames[j])
        coldata = data[!, j]
        i   = findfirst(==('_'), colname)
        age = parse(Int, colname[(i+1):end])
        n   = sum(x for x in coldata if !ismissing(x))  # The last row has missing data
        push!(result, (age=age, count=n))
        ntotal += n
    end
    result[!, :proportion] = result.count ./ ntotal
    result
end

################################################################################
# Script

# Get data
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\asgs_codes.tsv"
codes  = DataFrame(CSV.File(infile; delim='\t'))
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\population_by_age_by_sa2.tsv"
data   = DataFrame(CSV.File(infile; delim='\t'))
data   = join(codes, data, on=:SA2_NAME_2016, kind=:left)
data   = data[data.STATE_NAME_2016 .== "Victoria", :]

# Format and write to disk
data    = age_distribution(data)  # Columns: age, count
data    = data[data.count .> 0, :]
outfile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\consumable\\population_by_age_VIC.tsv"
CSV.write(outfile, data; delim='\t')
