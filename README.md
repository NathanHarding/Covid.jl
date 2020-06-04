# Covid.jl

Read the docs [here](docs/Overview.md)

The model can be used without having to write any code.

Configure the model in `${path to Covid.jl}/config/config.yml`

Then from your terminal: `julia ${path to Covid.jl}/scripts/runmodel.jl`

Or from the REPL:

```julia
repodir = "/path/to/Covid.jl"
using Pkg
Pkg.activate(repodir)  # Point Julia to "/path/to/Covid.jl/Manifest.toml"
Pkg.instantiate()      # Fetch and build any dependencies you don't already have
using Covid            # Load the code
Covid.runmodel(joinpath(repodir, "config", "config.yml"))
```

To train a model, specify the parameters and/or policies to be estimated.
See `config/train.yml` for an example.
Then run:

```julia
repodir = "/path/to/Covid.jl"
using Pkg
Pkg.activate(repodir)  # Point Julia to "/path/to/Covid.jl/Manifest.toml"
Pkg.instantiate()      # Fetch and build any dependencies you don't already have
using Covid            # Load the code
Covid.trainmodel(joinpath(repodir, "config", "train.yml"))
```