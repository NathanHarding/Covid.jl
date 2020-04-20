cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using CSV
using Covid
using DataFrames
using Dates

logit(p::Float64) = log(p / (1.0 - p))

# Params
npeople = 200
maxtime = 180
params  = (b0_E=log(7), b1_E=0.01, b0_I=log(14), b1_I=0.01, b0_H=log(3), b1_H=0.01, b0_C=log(4), b1_C=0.01, b0_V=log(4), b1_V=0.01,
           a0_H=logit(0.2), a1_H=0.01, a0_C=logit(0.25), a1_C=0.01, a0_V=logit(0.5), a1_V=0.01, a0_D=logit(0.8), a1_D=0.01,
           a0_infect=logit(0.99), a1_infect=0.01)
dist0   = [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SEIHCVRD

# Run
#npeople = 1_000_000
model = Covid.model5.init_model(npeople, params, maxtime, dist0);
println(now())
data  = run!(model);
println(now())

# Write to disk
CSV.write("C:\\projects\\data\\dhhs\\covid-abm\\model5.tsv", data; delim='\t')

#=
using ProfileView

# Trigger compilation
model = Covid.model5.init_model(npeople, params, maxtime, dist0);
println(now())
@profview data = run!(model);
println(now())

# Profile
npeople = 1_000_000
model = Covid.model5.init_model(npeople, params, maxtime, dist0);
println(now())
@profview data = run!(model);
println(now())

=#