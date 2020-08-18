base_wdir = "D:/ABM_test"
julia_install_directory = "C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia"
first_run = TRUE

if(first_run){
setwd(file.path(base_wdir,"Demographics.jl"))
system(paste(julia_install_directory, "scripts\\generate_population_files.jl"))
setwd("test")
system(paste(julia_install_directory, "runtests.jl"))
}


setwd(file.path(base_wdir,"Covid.jl"))
system("C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia scripts\\runmodel.jl")