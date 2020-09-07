import Pkg
@info pwd()
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()
using Covid
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--folder_name","-f"
            help = "name of output subfolders"
        "nconfigs"
            help = "number of config files per series run"
            arg_type = Int
            required = true
    end

    return parse_args(s)
end


parsed_args = parse_commandline()
show(parsed_args)
for i = 0:parsed_args["nconfigs"]-1
	cfile = "config_" * parsed_args["folder_name"] *"_" * string(i) * ".yml"
	Covid.runmodel(joinpath(pwd(), "config", cfile))
	Covid.GSAwrite(joinpath(pwd(), "config", cfile))
end





end