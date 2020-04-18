module Covid

include("model1.jl")
include("model2.jl")
include("model3.jl")

using .model1
using .model2
using .model3

end
