using Pkg
Pkg.activate(pwd())
using Covid
Covid.main(joinpath(pwd(), "config", "config.yml"))