
using Pkg
Pkg.activate(".")
using DataFrames
using CSV
using Dates

"""
Script for generating time series and summary statistics on the PHN level in Victoria
"""

#functions
function gen_dict(data,PHN)
	SA2toPHN = Dict{Int,String}()
	for location in unique(data.address)
		if location !=0 #ignore when data relates to all locations
			i = findfirst(x->x == location,skipmissing(PHN.SA2_MAINCODE_2016))
			@info i
			@info location
			SA2toPHN[PHN.SA2_MAINCODE_2016[i]] = PHN.cluster_name[i]
		end
	end
	SA2toPHN[0] = "Total"
	SA2toPHN
end

function get_PHN_names(PHN)
	PHN_names_VIC = Vector{String}()
	for i in unique(skipmissing(PHN.cluster_name))
		push!(PHN_names_VIC,i)
	end
	sort!(PHN_names_VIC)
	@info PHN_names_VIC
end

function filter2PHN!(data,PHN)
	SA2toPHN = gen_dict(data,PHN)
	PHN_names_VIC = get_PHN_names(PHN)
	tmp = Vector(undef,length(data.address))
	for (i,a) in enumerate(data.address)
		tmp[i] = SA2toPHN[a]
	end
	data.PHN = tmp
	@info "Aggregating data"
	gdf = groupby(data,[:run,:date,:PHN])
	result = aggregate(gdf,sum)
	result
end

function filter2HS!(data,PHN)
	SA2toPHN = gen_dict(data,PHN)
	PHN_names_VIC = get_PHN_names(PHN)
	tmp = Vector(undef,length(data.address))
	for (i,a) in enumerate(data.address)
		tmp[i] = SA2toPHN[a]
	end
	data.PHN = tmp
	@info "Aggregating data"
	gdf = groupby(data,[:run,:date,:PHN])
	result = aggregate(gdf,sum)
	result
end


#script
infile = ".\\data\\output\\metrics.csv"
hn_data = ".\\data\\input\\sa2_to_cluster.tsv" 

@info "Reading in data files"
data = DataFrame(CSV.File(infile))
PHN = DataFrame(CSV.File(hn_data))
@info "Reformatting data"
data = filter2PHN!(data,PHN)

CSV.write(".\\data\\output\\metrics_health_services.csv",data)
