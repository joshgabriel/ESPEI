from espei.espei_script import get_run_settings, run_espei
import os

if __name__ == '__main__':

   root_path = os.path.abspath(os.path.curdir)
   my_prior = [{'name': 'triangular', 'loc_shift_relative': -10.0, 'scale_relative': 20.0,'c':0.5} for i in list(range(0,15))]
   input_dict = {
    'system': {
        'phase_models': root_path + '/Cu-Mg_phases.json',
        'datasets': root_path + '/param_gen_data'
    },
    'mcmc': {
        'iterations': 1,
        'input_db': root_path + '/UnarySetScreening/Cu-Mg_gen_CuNUM.tdb',
        'scheduler': 'dask',
        'prior': my_prior,
        'restart_trace':'trace_zpf.npy',
        'save_interval':1 
    },
    'output':{
        'output_db': root_path + '/Cu-Mg_mcmc_zpf-1.tdb',
        'tracefile':'trace_prop.npy',
        'probfile':'prob_prop.npy',
        'logfile':'mcmc.log',
        'verbosity':2
    }
   }

   run_espei(get_run_settings(input_dict))
