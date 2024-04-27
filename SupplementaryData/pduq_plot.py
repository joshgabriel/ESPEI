import numpy as np
import seaborn as sns
from pycalphad import Database, variables as v
from pduq.dbf_calc import eq_calc_samples
from pduq.uq_plot import plot_phasereg_prob, plot_superimposed
from espei.utils import ImmediateClient
from dask.distributed import Client
from distributed.deploy.local import LocalCluster
import time
import linecache 
import sys
import os 
import pickle
import xarray

def print_exception():
    """
    Error exception catching function for debugging
    can be a very useful tool for a developer
    move to utils and activate when debug mode is on
    """
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno,
                                                       line.strip(), exc_obj))
if __name__ == '__main__':
   for num,iters in enumerate([1]):
      try:
        #os.mkdir("Iter")
        perc_params = []
        client = ImmediateClient(scheduler_file='scheduler.json')
        dbf = Database('UnarySetScreening/Cu-Mg_gen_Cu0.tdb') 
        all_params = np.load('trace_370.npy')
        #sel_params = np.load('ChooseSamplesExactCoOptimizeToCompareWithPropagate0.6K_Cu.npy')
        params = np.array([all_params[s,-1,:] for s in list(range(0,30))])

        conds = {v.P: 101325, v.T: (990, 1020, 10), v.X('MG'): (0.0, 1.0, 0.01)}

        eq = eq_calc_samples(dbf, conds, params,client=client,savef="Propagated_1000K_30Params_0_6K.pkl")
      except:
         print (print_exception())
