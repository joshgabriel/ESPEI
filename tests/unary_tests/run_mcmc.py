from espei.espei_script import get_run_settings, run_espei
import os

if __name__ == '__main__':

   root_path = os.path.abspath(os.path.curdir)
   binary_priors = [{'name': 'triangular', 'loc_shift_relative': -5.0, 'scale_relative': 10.0,'c':0.5} for i in list(range(0,15))]
   unary_priors = [{'name':'uniform', 'loc_shift_relative':-5.0, 'scale_relative': 10.0} for i in list(range(0,7))]
   my_prior = binary_priors + unary_priors
   
   input_dict = {
    'system': {
        'phase_models': root_path + '/Cu-Mg_phases.json',
        'datasets': root_path + '/param_gen_data',
        'unary': {'data': root_path + '/JSON_data_copper',
                  'param_spec': {'Param_A_range':[15,21],
                                 'Tm_A_range':[1300.0,1400.0]}
                 } 
    },
    'mcmc': {
        'iterations': 500,
        'input_db': root_path + '/Cu-Mg_gen.tdb',
        'scheduler': 'dask',
        'prior': my_prior,
        'chains_per_parameter': 10,
        'chain_std_deviation': 0.05,
        'save_interval':1 
    },
    'output':{
        'output_db': root_path + '/Cu-Mg_mcmc_zpf-2.tdb',
        'tracefile':'trace_prop.npy',
        'probfile':'prob_prop.npy',
        'logfile':'mcmc.log',
        'verbosity':2
    }
   }

   run_espei(get_run_settings(input_dict))
