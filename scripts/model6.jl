cd("C:\\projects\\repos\\Covid.jl")
using Pkg
Pkg.activate(".")
using Covid
configfile = "config\\config.yml"
Covid.main(configfile)