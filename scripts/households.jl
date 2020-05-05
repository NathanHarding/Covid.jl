#=
  Contents: Script for constructing households by SA2.
  The intermediate data is copied from the raw Excel spreadsheet sourced from the ABS.
=#

cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using CSV
using DataFrames

################################################################################
# Functions

"""
Move couples without kids from 2-person family households to 2-person non-family households.

This facilitates populating households with children before households without children.
"""
function move_couples!(data::DataFrame)
    n = size(data, 1)
    couplenokids = Symbol("One family household: Couple family with no children")
    data[!, :couplenokids] = missings(Int, n)
    for i = 1:n
        ismissing(data[i, couplenokids]) && continue
        ismissing(data[i, :Num_Psns_UR_2_FamHhold])    && continue
        ismissing(data[i, :Num_Psns_UR_2_NonFamHhold]) && continue
        n_couplenokids = min(data[i, :Num_Psns_UR_2_FamHhold], data[i, couplenokids])
        data[i, :Num_Psns_UR_2_FamHhold]    -= n_couplenokids
        data[i, :Num_Psns_UR_2_NonFamHhold] += n_couplenokids
    end
    select!(data, Not(couplenokids))
end

"Columns: nresidents, nhouseholds, proportion"
function construct_families(hholds::DataFrame)
    result = DataFrame(nresidents=Int[], nhouseholds=Int[])
    push!(result, (nresidents=2, nhouseholds=sum(hholds.Num_Psns_UR_2_FamHhold)))
    push!(result, (nresidents=3, nhouseholds=sum(hholds.Num_Psns_UR_3_FamHhold)))
    push!(result, (nresidents=4, nhouseholds=sum(hholds.Num_Psns_UR_4_FamHhold)))
    push!(result, (nresidents=5, nhouseholds=sum(hholds.Num_Psns_UR_5_FamHhold)))
    push!(result, (nresidents=6, nhouseholds=sum(hholds.Num_Psns_UR_6mo_FamHhold)))
    nh = sum(result.nhouseholds)
    result[!, :proportion] = result.nhouseholds ./ nh
    result
end

"Columns: nadults, nhouseholds, proportion"
function construct_nonfamilies(hholds::DataFrame)
    result = DataFrame(nadults=Int[], nhouseholds=Int[])
    push!(result, (nadults=1, nhouseholds=sum(hholds.Num_Psns_UR_1_NonFamHhold)))
    push!(result, (nadults=2, nhouseholds=sum(hholds.Num_Psns_UR_2_NonFamHhold)))
    push!(result, (nadults=3, nhouseholds=sum(hholds.Num_Psns_UR_3_NonFamHhold)))
    push!(result, (nadults=4, nhouseholds=sum(hholds.Num_Psns_UR_4_NonFamHhold)))
    push!(result, (nadults=5, nhouseholds=sum(hholds.Num_Psns_UR_5_NonFamHhold)))
    push!(result, (nadults=6, nhouseholds=sum(hholds.Num_Psns_UR_6mo_NonFamHhold)))
    nh = sum(result.nhouseholds)
    result[!, :proportion] = result.nhouseholds ./ nh
    result
end

################################################################################
# Script

# Get data
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\asgs_codes.tsv"
codes  = DataFrame(CSV.File(infile; delim='\t'))
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\family_household_composition.tsv"
data   = DataFrame(CSV.File(infile; delim='\t'))
data   = join(codes, data, on=:SA2_NAME_2016, kind=:left)
data   = data[data.STATE_NAME_2016 .== "Victoria", [:SA2_MAINCODE_2016, Symbol("One family household: Couple family with no children")]]
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\household_size.tsv"
hholds = DataFrame(CSV.File(infile; delim='\t'))

# Move couples without kids from 2-person family households to 2-person non-family households
hholds = join(hholds, data, on=:SA2_MAINCODE_2016, kind=:left)
move_couples!(hholds)
families    = construct_families(hholds)     # Columns: nresidents, nhouseholds, proportion
nonfamilies = construct_nonfamilies(hholds)  # Columns: nadults,    nhouseholds, proportion
outfile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\consumable\\family_households.tsv"
CSV.write(outfile, families; delim='\t')
outfile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\consumable\\nonfamily_households.tsv"
CSV.write(outfile, nonfamilies; delim='\t')
