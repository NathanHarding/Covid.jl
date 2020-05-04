#=
  Contents: Script for converting intermediate work place data to consumable data.
  The intermediate data is copied from the raw Excel spreadsheet sourced from the ABS.
=#

cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using CSV
using DataFrames

################################################################################
# Functions

function count_employees(data, colnames)
    result = Dict{Tuple{Int, Int}, Int}()  # nemployees_lb => count
    for colname in colnames
        s  = string(colname)
        i1 = findfirst(==('_'), s) + 1
        i2 = findfirst("to", s)
        i2 = isnothing(i2) ? findfirst("plus", s) : i2
        i2 = isnothing(i2) ? i1 : i2[1] - 1
        lb = parse(Int, s[i1:i2])
        if occursin("to", s)
            i1 = findfirst("to", s)[end] + 1
            ub = parse(Int, s[i1:end])
        elseif occursin("plus", s)
            ub = 5_000
        else
            ub = 0
        end
        result[(lb, ub)] = sum(data[!, colname])
    end
    result
end

function counts_to_table(grp2count::Dict)
    result = DataFrame(nemployees_lb=Int[], nemployees_ub=Int[], count=Int[])
    ntotal = 0
    for (lb_ub, n_businesses) in grp2count
        d = (nemployees_lb=lb_ub[1], nemployees_ub=lb_ub[2], count=n_businesses)
        push!(result, d)
        ntotal += n_businesses
    end
    result[!, :proportion] = result.count ./ ntotal
    sort!(result, :nemployees_lb)
    result
end

################################################################################
# Script

# Get data
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\workplace_size_by_industry_by_SA2.tsv"
data   = DataFrame(CSV.File(infile; delim='\t'))

# Exclude non-VIC data
data.SA2_code = [ismissing(x) ? missing : string(x) for x in data.SA2_code]
keep = [!ismissing(x) && x[1] == '2' for x in data.SA2_code]
data = data[keep, :]
keep = nothing

# Count groups
colnames  = [:nemployees_0, :nemployees_1to4, :nemployees_5to19, :nemployees_20to199, :nemployees_200plus]
grp2count = count_employees(data, colnames)
# Check: sum(0.5 * (k[1] + k[2] + 1)*v for (k,v) in grp2count)  # Should be around 6,400,000 (Vic population)

# Construct result and write to disk
data    = counts_to_table(grp2count)
outfile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\consumable\\workplace_by_size_VIC.tsv"
CSV.write(outfile, data; delim='\t')
