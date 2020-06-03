using Pkg
Pkg.activate(pwd())
using Covid
Covid.runmodel(joinpath(pwd(), "config", "config.yml"))