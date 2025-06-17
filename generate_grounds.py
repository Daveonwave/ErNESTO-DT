import numpy as np
from joblib import Parallel, delayed
from ernesto.utils.logger import CustomFormatter
from ernesto.digital_twin.orchestrator.orchestrator import DTOrchestrator
from ernesto import get_args, run_experiment, parse_submodels
from ernesto.preprocessing.schema import read_yaml
from rich import pretty
from copy import deepcopy


# Code to generate groun truth data for online learning using different ambient temperatures
# We are not gonna use this script for the moment, but it is useful to have it in case we need more realistic experiments
if __name__ == '__main__':
    args = get_args()
    parse_submodels(args)
    
    # Parallel execution of the experiments
    config_file = read_yaml(yaml_file=args['config_files'][0], yaml_type=args['mode'])
    
    # Define the different ambient temperatures 
    parallel_exp_config = []
    temps = np.arange(0.0, 42.5, 2.5)
    
    for i, temp in enumerate(temps):   
        config_file['destination_folder'] = 'ground_t_ambs/t_amb-{}_{}'.format(str.split(str(temp), '.')[0], str.split(str(temp), '.')[1])
        config_file['input']['ground_data']['vars'][1]['label'] = str(temp)
        config_file['battery']['init']['temperature'] = temp + 273.15
        parallel_exp_config.append(deepcopy(config_file))
    
    # Set the flag to avoid reading the config file again
    args['already_read'] = True
    
    n_cores = 9
    
    if n_cores == 1:
        run_experiment(args, parallel_exp_config[0])
    else:
        Parallel(n_jobs=n_cores)(delayed(run_experiment)(args, config) for config in parallel_exp_config)

