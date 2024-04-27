import os
import numpy as np
from pycalphad import Database
from espei.analysis import truncate_arrays
from espei.utils import database_symbols_to_fit, optimal_parameters

from pycalphad import binplot, variables as v
from espei.datasets import load_datasets, recursive_glob
from espei.plot import dataplot
import matplotlib.pyplot as plt
from glob import glob

INPUT_TDB_FILENAME = "Cu-Mg_mcmc_+Cu_2+.tdb"#"UnarySetScreening/Cu-Mg_gen_Cu0.tdb"
TRACE_FILENAME = "trace_+Cu_2+.npy"#"trace_370.npy"
ops = np.load(TRACE_FILENAME)[:,-1,:]
for n,p in enumerate([ops[1]]):
    dbf = Database(INPUT_TDB_FILENAME)
    print ("Taken input")
    plot_params = dict(zip(database_symbols_to_fit(dbf), p))
    dbf.symbols.update(plot_params)
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'CUMG2', 'LAVES_C15']
    conds = {v.P: 101325, v.T: (300, 1370, 1), v.X('MG'): (0, 1, 0.01)}

    # plot the phase diagram and data
    plot_kwargs = {"tielines":False, "tieline_color":(0, 1, 0, 1), "scatter":True}
    
    ax = binplot(dbf, comps, phases, conds, plot_kwargs=plot_kwargs)
    plt.close()
    
    #os.system("cp pd_data.csv PDplots/pd_dataCu4K_p100_index147_149_sanity.csv".format(n))

