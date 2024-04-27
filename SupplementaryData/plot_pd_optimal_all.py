import os
import numpy as np
from espei.utils import database_symbols_to_fit
from pycalphad import Database
from pycalphad import binplot, variables as v
import numpy as np
import time 

INPUT_TDB_FILENAME = "Cu-Mg_mcmc_zpf-40.tdb"#"TDB_sources/Cu-Mg_gen_Cu{}.tdb"
dbf = Database(INPUT_TDB_FILENAME)
params = np.load('trace_prop40.npy')
probs = np.load('prob_prop40.npy')[:,-1]
sel_index = [n for n,p in enumerate(probs) if p!=-np.inf]
for i in sel_index:
    p = params[i,-1,:]
    t1 = time.time()
    plot_params = dict(zip(database_symbols_to_fit(dbf), p)) 
    dbf.symbols.update(plot_params)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'CUMG2', 'LAVES_C15']
    conds = {v.P: 101325, v.T: (300, 1400, 10), v.X('MG'): (0, 1, 0.01)}

    plot_kwargs = {"tielines":False, "tieline_color":(0, 1, 0, 1), "scatter":True}
    
    ax = binplot(dbf, comps, phases, conds, plot_kwargs=plot_kwargs)
    
    os.system("cp pd_data.csv PDplots10K_10K_p40/pd_data_Cu{0}.csv".format(i))
    t2 = time.time()
    print (t2-t1)
