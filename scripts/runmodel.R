base_wdir = "D:/ABM_test"
julia_install_directory = "C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia"

setwd(file.path(base_wdir,"Covid.jl"))
file.create("data/output/GSA.csv")
header = paste("Recovered","Positives", "seed","dist_hh","dist_sch","dist_w",
               "dist_com","dist_soc","test_IS","test_W","trac_hh_s","trac_hh_a",
               "trac_sch_s","trac_sch_a","trac_w_s","trac_w_a","",sep = ",")
ofile <- file("data/output/GSA.csv")
writeLines(header, ofile)

#load function to write config files
source('scripts/write_config.R')

#set parameters, in 'if' just to run as one line
if(1){
first_day = "2020-01-01"
last_day = "2020-01-15"

nruns = 1

seed_dates <- c("2020-01-01")
seed_vals <- c(10)

dist_dates <- c("2020-01-01")
dist_vals <- 
  c(0.5,0.5,0.5,0.5,0.5)

test_dates <- c("2020-01-01")
test_vals <- c(0.1,0.9)

trace_dates <- c("2020-01-01")
trace_vals <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

quar_dates <- c("2020-01-01")
quar_days <-c(1,2,3,4,5,6)
quar_compl <-c(0.9,0.8,0.7,0.6,0.5,0.4)
}


for (i in 1:1){
#update_parameters  
dist_vals <- dist_vals + runif(length(dist_vals))* pmin(dist_vals,abs(dist_vals-1))
test_vals <- test_vals + runif(length(test_vals))* pmin(test_vals,abs(test_vals-1))
trace_vals <- trace_vals + runif(length(trace_vals))* pmin(trace_vals,abs(trace_vals-1))
quar_compl <- quar_compl + runif(length(quar_compl))* pmin(quar_compl,abs(quar_compl-1))
#write config file
write_config_f(first_day, last_day, nruns, seed_dates, seed_vals,dist_dates, dist_vals,
               test_dates, test_vals, trace_dates, trace_vals, 
               quar_dates, quar_days, quar_compl)

#run model
system(paste(julia_install_directory, "scripts\\runmodel.jl"))
system(paste(julia_install_directory, "scripts\\filter_results.jl"))
# filter more to reduce size of metrics_filtered.csv (for sensitivity want to know about number of infected and )
df <- read.csv("data/output/metrics_filtered.csv")
df <- df[df$date == last_day, ]
actual_infections = mean(df$R)
observed_cases = mean(df$positives)

writer = paste(c(actual_infections,observed_cases, seed_vals,dist_vals,test_vals,trace_vals,quar_days,quar_compl))
writer <- as.matrix(t(writer))
ofile <- file("data/output/GSA.csv",'a')
write.table(writer, file = ofile,sep = ",",col.names = FALSE,append = FALSE)
close(ofile)
}