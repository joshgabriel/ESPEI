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
import time
import pandas as pd

for n,nam in enumerate(glob("*mcmc*+.tdb")):
    #print (nam)
    tr_nam = nam.replace('Cu-Mg_mcmc','trace').replace('.tdb','.npy')
    
    t1 = time.time()
    params = np.load(tr_nam)[:,-1,:]
    dbf = Database(nam)
    plot_params = dict(zip(database_symbols_to_fit(dbf), params[-3]))
    comps = ['CU', 'MG', 'VA']
    phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'CUMG2', 'LAVES_C15']
    conds = {v.P: 101325, v.T: (300, 1370, 10), v.X('MG'): (0, 1, 0.01)}

    # plot the phase diagram and data
    plot_kwargs = {"tielines":False, "tieline_color":(0, 1, 0, 1), "scatter":True}
    
    ax = binplot(dbf, comps, phases, conds, plot_kwargs=plot_kwargs)
    #plt.close()
    
    os.system("cp pd_data.csv PDplots/pd_dataCu4K_prop_{0}_i147.csv".format(nam))
    print (params.shape, nam,tr_nam,pd.read_csv('pd_data.csv')['x'].max())
    t2 = time.time()
    print (t2-t1)
