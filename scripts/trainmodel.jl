using Pkg
Pkg.activate(pwd())
using Covid
Covid.trainmodel(joinpath(pwd(), "config", "train.yml"))