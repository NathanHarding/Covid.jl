module Covid

include("run_model.jl")  # Exports: runmodel
using .run_model

#include("train_model.jl")  # Exports: trainmodel
include("train_model_abc.jl")  # Exports: trainmodel
using .train_model

end
