cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using Agents  # step!
using CSV
using Covid
using Covid.model3
using DataFrames
using Dates

# Params
griddims = (10, 10)
npeople  = 200
prms     = Dict{Symbol, Any}(:dur_exposed => 7, :dur_infectious => 14, :dur_hospitalised => 3, :dur_icu => 4, :dur_ventilated => 4,
                             :p_infect => 1.0, :p_hospital => 0.2, :p_icu => 0.25, :p_ventilation => 0.5, :p_death => 0.8, :p_reinfection => 0.0,
                             :age => 20)
dist0    = [0.98, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # SEIHCVRD

# Data collection
nsteps = 180
when   = 1:nsteps  # Collect data at these steps (including step 0)
nsusceptible(x)  = count(i == :S for i in x)
nexposed(x)      = count(i == :E for i in x)
ninfected(x)     = count(i == :I for i in x)
nhospitalised(x) = count(i == :H for i in x)
nicu(x)          = count(i == :C for i in x)
nventilated(x)   = count(i == :V for i in x)
nrecovered(x)    = count(i == :R for i in x)
ndeceased(x)     = count(i == :D for i in x)
props  = Dict(:status => [nsusceptible, nexposed, ninfected, nhospitalised, nicu, nventilated, nrecovered, ndeceased])

# Run
model = model3.init_model(griddims, npeople, prms, dist0, model3.Conductor(nsteps, model3.EventBag));
println(now())
data  = step!(model, model3.agent_step!, model3.model_step!, nsteps, props, when=when);
println(now())

# Write to disk
CSV.write("C:\\projects\\data\\dhhs\\covid-abm\\model3.tsv", data; delim='\t')

#=
griddims = (500, 500)
npeople  = 1_000_000
model = model3.init_model(griddims, npeople, prms, dist0, model3.Conductor(nsteps, model3.EventBag));
println(now())
data  = step!(model, model3.agent_step!, model3.model_step!, nsteps, props, when=when);
println(now())


using ProfileView

# Trigger compilation
griddims = (10, 10)
npeople  = 200
model    = model3.init_model(griddims, npeople, prms, dist0, model3.Conductor(nsteps, model3.EventBag));
println(now())
@profview data = step!(model, model3.agent_step!, model3.model_step!, nsteps, props, when=when);
println(now())

# Profile
griddims = (500, 500)
npeople  = 1_000_000
model    = model3.init_model(griddims, npeople, prms, dist0, model3.Conductor(nsteps, model3.EventBag));
println(now())
@profview data = step!(model, model3.agent_step!, model3.model_step!, nsteps, props, when=when);
println(now())

=#