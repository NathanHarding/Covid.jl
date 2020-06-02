module Covid

include("runmodel.jl")  # Exports: run
using .runmodel

include("trainmodel.jl")  # Exports: train!
using .trainmodel

end
