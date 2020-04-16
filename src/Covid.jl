module Covid

include("model1.jl")
include("model2.jl")

using .model1
using .model2

end
