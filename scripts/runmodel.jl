using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using Covid
Covid.runmodel(joinpath(pwd(), "config", "config.yml"))