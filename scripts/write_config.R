indent <- "    "
first_day = "2020-01-01"
last_day = "2020-04-15"
nruns = 10

seed_dates <- c("2020-01-01","2020-02-03")
seed_vals <- c(10,20)

dist_dates <- c("2020-01-01","2020-02-03")
dist_vals <- c(c(0.5,0.5,0.5,0.5,0.5),c(0.1,0.2,0.3,0.4,0.5))

test_dates <- c("2020-01-01","2020-02-03")
test_vals <- c(c(0.1,0.9),c(0.2,0.8))

trace_dates <- c("2020-01-01")
trace_vals <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0)

quar_dates <- c("2020-01-01")
quar_days <-c(1,2,3,4,5,6)
quar_compl <-c(0.9,0.8,0.7,0.6,0.5,0.4)

preamble <- "demographics_datadir: \"../Demographics.jl/test/data/output\"  # Directory containing saved population data. people = load(demographics_datadir)\noutput_directory: \"data/output\"  # Full path is /path/to/Covid.jl/data/output"
preamble <- paste(preamble,"\nfirstday:",first_day)
preamble <- paste(preamble,"\nlastday:",last_day)
preamble<- paste(preamble,"\nnruns:",nruns)
preamble<- paste(preamble,"\nparams: \"data/input/params.tsv\"  # Full path is /path/to/Covid.jl/data/input/params.tsv
cumsum_population: \"../Demographics.jl/test/data/input/population_by_SA2.tsv\"")


forcing<- "\n\nforcing:"
for (i in 1:length(seed_dates))
{
  forcing<-paste0(forcing,"\n",indent,seed_dates[i],": {Any : {E: ",seed_vals[i],"}}")
}

distancing <- "\n\ndistancing_policy:"
for (i in 1:length(dist_dates))
{
  distancing<-paste0(distancing,"\n",indent,dist_dates[i],
                     ": {household: ",dist_vals[5*i-4],
                     ", school: ",dist_vals[5*i-3],
                     ", workplace: ",dist_vals[5*i-2],
                     ", community: ",dist_vals[5*i-1],
                     ", social: ", dist_vals[5*i],"}")
}
testing <- "\n\ntesting_policy:"
for (i in 1:length(test_dates))
{
  testing<-paste0(testing,"\n",indent,test_dates[i],
                  ": {IS: ",test_vals[2*i-1],
                  ", W: ",test_vals[2*i],"}")
}
tracing <- "\n\ntracing_policy:"
for (i in 1:length(trace_dates))
{
  tracing<-paste0(tracing,"\n",indent,trace_dates[i],
                  ": {household: {symptomatic: ",trace_vals[10*i-9],
                  ", asymptomatic: ", trace_vals[10*i-8],
                  "}, school: {symptomatic: ",trace_vals[10*i-7],
                  ", asymptomatic: ", trace_vals[10*i-6],
                  "}, workplace: {symptomatic: ",trace_vals[10*i-5],
                  ", asymptomatic: ", trace_vals[10*i-4],
                  "}, community: {symptomatic: ",trace_vals[10*i-3],
                  ", asymptomatic: ", trace_vals[10*i-2],
                  "}, social: {symptomatic: ", trace_vals[10*i-1],
                  ", asymptomatic: ", trace_vals[10*i],"}}")
}

quarantine <- "\n\nquarantine_policy:"
for (i in 1:length(quar_dates))
{
  quarantine<-paste0(quarantine,"\n",indent,quar_dates[i],
                     ": {awaiting_test_result: {days: ",quar_days[6*i-5],
                     ", compliance: ", quar_compl[6*i-5],
                     "}, tested_positive: {days: ",quar_days[6*i-4],
                     ", compliance: ", quar_compl[6*i-4],
                     "}, case_contacts: ",
                     "{workplace: {days: ",quar_days[6*i-3],
                     ", compliance: ", quar_compl[6*i-3],
                     "}, school: {days: ",quar_days[6*i-2],
                     ", compliance: ", quar_compl[6*i-2],
                     "}, community: {days: ",quar_days[6*i-1],
                     ", compliance: ", quar_compl[6*i-1],
                     "}, social: {days: ",quar_days[6*i],
                     ", compliance: ", quar_compl[6*i],"}}}")
}
config = paste(preamble,forcing,distancing,testing,tracing,quarantine)
write(config,"../config/config.yml")

