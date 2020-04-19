cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using CSV
using Covid
using Covid.model4
using DataFrames
using Dates

# Params
npeople = 200
maxtime = 180
params  = (mean_dur_E=7.0, mean_dur_I=14.0, mean_dur_H=3.0, mean_dur_C=4.0, mean_dur_V=4.0,
           p_infect=1.0, p_H=0.2, p_C=0.25, p_V=0.5, p_death=0.8, p_reinfection=0.0)
dist0   = [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SEIHCVRD

# Run
#npeople = 1_000_000
model = model4.init_model(npeople, params, maxtime, dist0);
println(now())
data  = model4.run!(model);
println(now())

# Write to disk
CSV.write("C:\\projects\\data\\dhhs\\covid-abm\\model4.tsv", data; delim='\t')

#=
using ProfileView

# Trigger compilation
model = model4.init_model(npeople, params, maxtime, dist0);
println(now())
@profview data  = model4.run!(model);
println(now())

# Profile
model = model4.init_model(npeople, params, maxtime, dist0);
println(now())
@profview data  = model4.run!(model);
println(now())

=#