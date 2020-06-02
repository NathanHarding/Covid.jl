using Pkg
Pkg.activate(pwd())
using Covid
Covid.train!(joinpath(pwd(), "config", "train.yml"))