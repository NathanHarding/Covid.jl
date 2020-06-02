using Pkg
Pkg.activate(pwd())
using Covid
Covid.run(joinpath(pwd(), "config", "config.yml"))