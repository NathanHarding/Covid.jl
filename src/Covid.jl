module Covid

export run!

include("core.jl")

using .core

#include("model1.jl")
#include("model2.jl")
#include("model3.jl")
include("model4.jl")

#using .model1
#using .model2
#using .model3
using .model4

end
