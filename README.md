# Covid.jl

Read the docs [here](docs/Overview.md)

The model can be used without having to write any code.

Configure the model in `${path to Covid.jl}/config/config.yml`

Then from your terminal: `julia ${path to Covid.jl}/scripts/runmodel.jl`

Or from the REPL:

```julia
repodir = "path to Covid.jl"
using Pkg
Pkg.activate(repodir)
using Covid
Covid.main(joinpath(repodir, "config", "config.yml"))
```