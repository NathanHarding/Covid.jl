# Running Covid ABM

In order to run Covid.jl from R you require the following:

## Julia install

You will need to download and install [Julia](https://julialang.org/downloads/).
When installing Julia take note of the install path, as we will need this to run Julia from the command line.
In particular we will require path\to\julia_X.Y\bin\julia

## Source files
The source files for this project are contained within two github repositories [Covid.jl](https://github.com/nathanharding/covid.jl) and [Demographics.jl](https://github.com/nathanharding/demographics.jl)
1. Clone both of this git repositories to the same parent directory. Something like ABM\Covid.jl and ABM\Demographics.jl.
2. Extract public_data_files_covid to the same parent directory.
	These directories need to share the same parent directory as many of the scripts use relative paths. Demographics.jl generates the population of agents including contacts whereas Covid.jl runs the epidemic simulation.

### Config files
in order to change the base probability of infection change
Covid.jl\\data\\input\\params.tsv 

In order to change the parameters of the model need to edit
Covid.jl\config\config.yml


## Running scripts
In order to run a julia file from the command line
Open a terminal
"path\to\julia_X.Y\bin\julia path\to\script.jl"

From R
system("path\to\julia_X.Y\bin\julia path\to\script.jl")

In order to run the main epidemic simulation

Open a terminal
"path\to\julia_X.Y\bin\julia path\to\Covid.jl\script\runmodel.jl"

This saves an output to "Covid.jl\data\output\metrics.csv". This output contains a time series for each run at the level of SA2. 
In order to aggregate to total values on each day we run Covid.jl\scripts\filter_results.jl which generates metrics_filtered.csv an aggregated time series.

### Running from R
- Open Covid.jl\\scripts\\runmodel.R
- Change base_wdir to the parent directory of Covid.jl, Demographics.jl and public_data_files_covid
- change julia_install_directory to your local julia install directory. By default it will be something like "C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia"
- If you don't want to regenerate the population (which takes around 10 minutes) set first_run = FALSE to skip population generation.

package installation should be taken care of by the julia scripts called by runmodel.R. If you have issues please email [me](nathan.harding@dhhs.vic.gov.au). These can generally be resolved by running once from the Julia REPL

### Running from the REPL
Open julia REPL
```julia
cd("path\to\parent\Demographics.jl")
]  #enters Pkg mode
activate .
st
instantiate
CTRL+C  # exit pkg mode
include("scripts\\generate_population_files.jl")
cd("test") 
include("runtests.jl")
```
-> runtests generates the population file ~ 650mb and takes about 10 minutes to run
```julia
cd path\to\parent\Covid.jl
]   # enters Pkg Mode
activate .
st
instantiate
CTRL+C   # exit pkg mode
include("scripts\\runmodel.jl")
```