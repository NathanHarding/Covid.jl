module Covid

export run!

include("config.jl")
include("utils.jl")
include("core.jl")
include("model6.jl")  # Depends on core
include("run.jl")

using .config
using .utils
using .core
using .model6
using .run

end
