using Pkg
Pkg.activate(pwd())
Pkg.instantiate()

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()
using Covid
Covid.runmodel(joinpath(pwd(), "config", "config.yml"))