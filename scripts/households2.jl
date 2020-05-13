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

function construct_household_counts(hh_size::DataFrame, hh_composition::DataFrame)
    hholds = join(hh_size, hh_composition, on=:SA2_MAINCODE_2016, kind=:left)
    dropmissing!(hholds)
    move_couples!(hholds)
    construct_result(hholds)
end

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

function construct_result(hholds)
    result = DataFrame(SA2_MAINCODE_2016=Int[], nadults=Int[], nchildren=Int[], nhouseholds=Int[])
    oneparent_families = Symbol("One family household: One parent family")
    for row in eachrow(hholds)
        # Households with children
        sa2code = row.SA2_MAINCODE_2016
        nresidents2households = construct_nresidents2households(row, oneparent_families)  # nresidents => (n_hholds_with_1_parent, n_hholds_with_2_parents)
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=1, nchildren=1, nhouseholds=row.Num_Psns_UR_2_FamHhold))
        for nresidents = 3:6
            n1, n2 = nresidents2households[nresidents]
            push!(result, (SA2_MAINCODE_2016=sa2code, nadults=1, nchildren=nresidents-1, nhouseholds=n1))
            push!(result, (SA2_MAINCODE_2016=sa2code, nadults=2, nchildren=nresidents-2, nhouseholds=n2))
        end

        # Households without children
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=1, nchildren=0, nhouseholds=row.Num_Psns_UR_1_NonFamHhold))
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=2, nchildren=0, nhouseholds=row.Num_Psns_UR_2_NonFamHhold))
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=3, nchildren=0, nhouseholds=row.Num_Psns_UR_3_NonFamHhold))
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=4, nchildren=0, nhouseholds=row.Num_Psns_UR_4_NonFamHhold))
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=5, nchildren=0, nhouseholds=row.Num_Psns_UR_5_NonFamHhold))
        push!(result, (SA2_MAINCODE_2016=sa2code, nadults=6, nchildren=0, nhouseholds=row.Num_Psns_UR_6mo_NonFamHhold))
    end
    result = result[result.nhouseholds .> 0, :]
    sort!(result, (:SA2_MAINCODE_2016, :nadults, :nchildren))
end

"Determine the number of 1-parent families with 3+ residents, and allocate them to families with 3+ residents."
function construct_nresidents2households(row, oneparent_families::Symbol)
    result = Dict{Int, Tuple{Int, Int}}()  # nresidents => (n_hholds_with_1_parent, n_hholds_with_2_parents)
    n_1parent_families = max(0, row[oneparent_families] - row.Num_Psns_UR_2_FamHhold)  # Assumes 2-person families with children have 1 parent
    nfamilies   = [row.Num_Psns_UR_3_FamHhold, row.Num_Psns_UR_4_FamHhold, row.Num_Psns_UR_5_FamHhold, row.Num_Psns_UR_6mo_FamHhold]
    ntotal      = sum(nfamilies)
    n1_unplaced = n_1parent_families
    for nresidents = 3:6  # Allocate 1-parent families to household sizes proportionally to the frequency of nresidents
        if n1_unplaced == 0
            result[nresidents] = (0, nfamilies[nresidents - 2])
            continue
        end
        n1 = nresidents < 6 ? round(Int, n_1parent_families * nfamilies[nresidents - 2] / ntotal) : n1_unplaced
        n1 = min(n1, n1_unplaced)
        n2 = max(0, nfamilies[nresidents - 2] - n1)
        n1_unplaced -= n1
        result[nresidents] = (n1, n2)
    end
    result
end

################################################################################
# Script

# Get input data
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\asgs_codes.tsv"
codes  = DataFrame(CSV.File(infile; delim='\t'))
infile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\family_household_composition.tsv"
hh_composition = DataFrame(CSV.File(infile; delim='\t'))
hh_composition = join(hh_composition, codes, on=:SA2_NAME_2016, kind=:left)
infile  = "C:\\projects\\data\\dhhs\\covid-abm\\input\\intermediate\\household_size.tsv"
hh_size = DataFrame(CSV.File(infile; delim='\t'))

# Construct result
households = construct_household_counts(hh_size, hh_composition)
outfile = "C:\\projects\\data\\dhhs\\covid-abm\\input\\consumable\\households.tsv"
CSV.write(outfile, households; delim='\t')
