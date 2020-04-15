cd("C:\\Users\\Owner\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")

using Agents  # step!
using Covid
using Covid.model1

# Params
npeople  = 200
griddims = (10, 10)
prms     = Dict{Symbol, Any}(:dur_exposed => 7, :dur_infectious => 21, :p_infect => 0.4, :p_death => 0.02, :p_reinfection => 0.01, :age => 20)
dist0    = [0.98, 0.02, 0.0, 0.0, 0.0]  # SEIRD

# Data collection
nsteps = 20
when   = 1:nsteps  # Collect data at these steps (including step 0)
nexposed(x)   = count(i == 'E' for i in x)
ninfected(x)  = count(i == 'I' for i in x)
nrecovered(x) = count(i == 'R' for i in x)
props  = Dict(:status => [nexposed, ninfected, nrecovered])

# Run
function model_step!(model)
    model.properties[:time] += 1
end
model = model1.init_model(griddims, npeople, prms, dist0; properties=Dict(:time => 0))
data  = step!(model, model1.agent_step!, model_step!, nsteps, props, when=when)
