using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
using Covid
Covid.trainmodel(joinpath(pwd(), "config", "train.yml"))