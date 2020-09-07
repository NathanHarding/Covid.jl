#----------------------------------------
#Generate config files for sensitivity analysis
#----------------------------------------

#----------------------------------------


base_wdir = "D:/ABM_test"
julia_install_directory = "C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia"
n_par_runs = 2
setwd(file.path(base_wdir,"Covid.jl"))
source("scripts\\write_config.R")

#----------------------------------------
# create array to store results
df <- data.frame(p_infect=numeric(), 
                 dp_hh=numeric(),
                 dp_sc=numeric(),
                 dp_wo=numeric(),
                 dp_co=numeric(),
                 dp_so=numeric(),
                 test_IS=numeric(),
                 test_W=numeric(),
                 trace_hh=numeric(),
                 trace_sc_w=numeric(),
                 trace_co_so=numeric(),
                 days_for_test=numeric(),
                 qp_atr=numeric(),
                 qp_tp=numeric(),
                 qp_cc=numeric()
                 ) 

#----------------------------------------
# create vectors with values for parameters
library(lhs)
q <- randomLHS(6, 15) 
colnames(q) <- c('p_infect',
                 'dp_hh',
                 'dp_sc',
                 'dp_wo',
                 'dp_co',
                 'dp_so',
                 'test_IS',
                 'test_W',
                 'trace_hh',
                 'trace_sc_w',
                 'trace_co_so',
                 'days_for_test',
                 'qp_atr',
                 'qp_tp',
                 'qp_cc')


dfq = data.frame(p_infect = q[,1],
                 dp_hh = q[,2],
                 dp_sc = q[,3],
                 dp_wo = q[,4],
                 dp_co = q[,5],
                 dp_so = q[,6],
                 test_IS = q[,7],
                 test_W = q[,8],
                 trace_hh = q[,9],
                 trace_sc_w = q[,10],
                 trace_co_so = q[,11],
                 days_for_test = floor(q[,12]*4+1),
                 qp_atr = q[,13],
                 qp_tp = q[,14],
                 qp_cc = q[,15])
#unused-probably useful
# par <- data.frame(p_infect = qunif(min=0.001, max=0.08),
#                   dp_hh = qunif(min = 0,max = 1),
#                   dp_sc = qunif(min = 0,max = 1),
#                   dp_wo = qunif(min = 0,max = 1),
#                   dp_co = qunif(min = 0,max = 1),
#                   dp_so = qunif(min = 0,max = 1),
#                   test_IS = qunif(min = 0,max = 1),
#                   test_W = qunif(min = 0,max = 1),
#                   trace_hh = qunif(min = 0,max = 1),
#                   trace_sc_w = qunif(min = 0,max = 1),
#                   trace_co_so = qunif(min = 0,max = 1),
#                   days_for_test = sample(1:6,1),
#                   qp_atr = qunif(min = 0,max = 1),
#                   qp_tp = qunif(min = 0,max = 1),
#                   qp_cc = qunif(min = 0,max = 1)
#                   )


for (row in 1:nrow(dfq)){
  outdir = paste0("output_",row%%n_par_runs)
  print(outdir)
  fname = paste0("config_",row%%n_par_runs,"_",(row-1)%/%n_par_runs)
  print(fname)
write_config_f("2020-01-01", "2021-01-01", 1,
               dfq[row,"p_infect"],
               "2020-01-01",c(100),
               "2020-01-01",c(dfq[row,"dp_hh"],dfq[row,"dp_sc"],dfq[row,"dp_wo"],dfq[row,"dp_co"],dfq[row,"dp_so"]),
               "2020-01-01",c(dfq[row,"test_IS"],dfq[row,"test_W"]),
               "2020-01-01",c(dfq[row,"trace_hh"],dfq[row,"trace_hh"],dfq[row,"trace_sc_w"],dfq[row,"trace_sc_w"],dfq[row,"trace_sc_w"],dfq[row,"trace_sc_w"],dfq[row,"trace_co_so"],dfq[row,"trace_co_so"],dfq[row,"trace_co_so"],dfq[row,"trace_co_so"]),
               "2020-01-01",c(dfq[row,"days_for_test"],10,14,14,14,14,14),
               c(dfq[row,"qp_atr"],dfq[row,"qp_tp"],dfq[row,"qp_cc"],dfq[row,"qp_cc"],dfq[row,"qp_cc"],dfq[row,"qp_cc"],dfq[row,"qp_cc"]),
               outdir,fname
               )
}


#Then run in the following way
#system("C:\\Users\\User\\AppData\\Local\\Programs\\Julia\\Julia-1.4.2\\bin\\julia scripts\\runmodel_series.jl 3 -f 1")
