using Covid

npeople  = 100
griddims = (10, 10)

nsteps = 5
when   = 1:5  # Collect data at these steps (including step 0)
props  = Dict(:status => [count])

prms  = Dict{Symbol, Any}(:dur_exposed => 7, :dur_infectious => 21, :infectiousness => 0.4, :p_death => 0.02, :p_reinfection => 0.0, :age => 20)
model = model1.init_model(griddims, npeople, prms; scheduler=fastest, properties=nothing)
data  = step!(model, model1.agent_step!, nsteps, props, when=when)