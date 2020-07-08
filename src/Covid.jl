module Covid

include("run_model.jl")  # Exports: runmodel
using .run_model

#include("train_model.jl")  # Exports: trainmodel
#include("train_model_abc.jl")  # Exports: trainmodel
#include("train_model_abc1p4.jl")  # Exports: trainmodel. Uses KissABC.jl version 1.4.
#include("train_model_abc_beaumont.jl")  # Exports: trainmodel
#include("train_model_sparsegrids.jl")  # Exports trainmodel
#include("train_model_cuhre.jl")  # Exports trainmodel
include("train_model_simplex.jl")
using .train_model

end
