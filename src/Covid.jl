module Covid

include("run_model.jl")  # Exports: runmodel
using .run_model

#include("train_model_optim.jl")
include("train_model_nlopt.jl")
using .train_model

end
