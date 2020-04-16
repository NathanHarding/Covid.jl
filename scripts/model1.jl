cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using Agents  # step!
using CSV
using Covid
using Covid.model1
using DataFrames

# Params
griddims = (10, 10)
npeople  = 200
prms     = Dict{Symbol, Any}(:dur_exposed => 7, :dur_infectious => 21, :p_infect => 1.0, :p_death => 0.02, :p_reinfection => 0.0)
dist0    = [0.98, 0.02, 0.0, 0.0, 0.0]  # SEIRD

# Data collection
nsteps = 180
when   = 1:nsteps  # Collect data at these steps (including step 0)
nsusceptible(x) = count(i == 'S' for i in x)
nexposed(x)     = count(i == 'E' for i in x)
ninfected(x)    = count(i == 'I' for i in x)
nrecovered(x)   = count(i == 'R' for i in x)
ndeceased(x)    = count(i == 'D' for i in x)
props  = Dict(:status => [nsusceptible, nexposed, ninfected, nrecovered, ndeceased])

# Run
function model_step!(model)
    model.properties[:time] += 1
end
model = model1.init_model(griddims, npeople, prms, dist0; properties=Dict(:time => 0))
data  = step!(model, model1.agent_step!, model_step!, nsteps, props, when=when);

# Write to disk
CSV.write("C:\\projects\\data\\dhhs\\covid-abm\\model1.tsv", data; delim='\t')
