import argparse
import logging
import sys
import os
from joblib import Parallel, delayed
from ernesto.utils.logger import CustomFormatter
from ernesto.digital_twin.orchestrator.orchestrator import DTOrchestrator


def run_experiment(args, config_file):
    args['config'] = config_file
    orchestrator = DTOrchestrator(**args)
    orchestrator.run()
    #orchestrator.evaluate()
    

def get_args():
    main_parser = argparse.ArgumentParser(description="Digital Twin of a Battery Energy Storage System (RSE)",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def get_driven_args():
        """
        Parser of arguments for DRIVEN SIMULATION mode
        """
        driven_parser.add_argument("--config_files", nargs='*', default=["./data/config/sim_config_example2.yaml"],
                                   help="Specifies the list of files containing parameters for each parallel experiment.")

    def get_schedule_args():
        """
        Parser of arguments for SCHEDULED SIMULATION mode
        """
        schedule_parser.add_argument("--config_files", nargs='*', default=["./data/config/scheduled_config.yaml"],
                                     help="Specifies the list of files containing parameters for each parallel experiment.")
        
        schedule_parser.add_argument("--iterations", default=500, type=int,
                                   help="Specifies the number of iterations of the entire experiment.")
        
        schedule_parser.add_argument("--timestep", default=1., type=float,
                                   help="Specifies the timestep of the simulator in seconds.")

    def get_adaptive_args():
        """
        Parser of arguments for ADAPTIVE SIMULATION mode
        """
        adaptive_parser.add_argument("--config_files", nargs='*', default=["./data/config/sim_adaptive.yaml"],
                                     help="Specifies the list of files containing parameters for each parallel experiment.")
        
        adaptive_parser.add_argument("--clusters_folder", default="./data/config/clusters/", type=str,
                                     help="Specifies the folder containing the clusters.")
        
        adaptive_parser.add_argument("--alpha", default=0.18, type=float,
                                  help="Specifies the regularization term of the loss function.")
        
        adaptive_parser.add_argument("--batch_size", default=10000, type=int,
                                  help="Specifies the size of the batch (window) used to perform an optimization of parameters.")
        
        adaptive_parser.add_argument("--alg", default='L-BFGS-B', type=str,
                                     help="Specifies the optimizer algorithm adopted.")
        
        adaptive_parser.add_argument("--n_restarts", default=1, type=int,
                                  help="Specifies the number of restarts for each optimization step.")

    def get_generic_args():
        """
        Arguments of the main parser that can be useful to all the kind of modes
        """
        electrical_choices = [os.path.splitext(file)[0] for file in os.listdir('./data/config/models/electrical')]
        thermal_choices = [os.path.splitext(file)[0] for file in os.listdir('./data/config/models/thermal')]
        aging_choices = [os.path.splitext(file)[0] for file in os.listdir('./data/config/models/aging')]
        
        main_parser.add_argument("--config_folder", action="store", default="./data/config", type=str,
                                 help="Specifies the folder which we retrieve preprocessing from.")

        main_parser.add_argument("--output_folder", action="store", default="./data/output", type=str,
                                 help="Specifies the name of the folder where to store the output results.")

        main_parser.add_argument("--ground_folder", action="store", default="./data/ground", type=str,
                                 help="Specifies the folder which we retrieve preprocessing from.")

        main_parser.add_argument("--assets", action="store", default="./data/config/assets.yaml",
                                 type=str, help="Specifies the file containing parameters useful for the experiment.")

        main_parser.add_argument("--electrical", nargs=1, choices=electrical_choices,
                                 help="Specifies the name of the mandatory (electrical) model of the battery. \
                                     It can be also a data-driven model. Models are stored in the folder 'models/electrical'.")

        main_parser.add_argument("--thermal", nargs=1, choices=thermal_choices,
                                 help="Specifies the name of the optional thermal model of the battery. \
                                     It can be also a data-driven model. Models are stored in the folder 'models/thermal'.")

        main_parser.add_argument("--aging", nargs=1, choices=aging_choices,
                                 help="Specifies the name of the optional aging model of the battery. \
                                     Models are stored in the folder 'models/aging'.")

        main_parser.add_argument("--n_cores", action="store", default=-1, type=int,
                                 help="Specifies the number of cores to use for parallel simulations. If save_results "
                                      "is set, cores will be override to 1 to limit RAM consumption.")
        
        main_parser.add_argument("--interactive", action="store_true",
                                 help="Enable the interaction of the user by a dedicated CLI.")

        
        main_parser.add_argument("--verbose", action="store_true",
                                 help="Increases logged information, but slows down the computation.")

    get_generic_args()
    subparsers = main_parser.add_subparsers(title="Mode", dest='mode', description="Experiment mode",
                                            help="Working mode of the Digital Twin", required=True)

    driven_parser = subparsers.add_parser('driven', help="Driven Simulation Mode",
                                          formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    get_driven_args()

    schedule_parser = subparsers.add_parser('scheduled', help="Scheduled Simulation Mode",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    get_schedule_args()

    adaptive_parser = subparsers.add_parser('adaptive', help="Adaptive Simulation Mode",
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    get_adaptive_args()

    main_args = vars(main_parser.parse_args())
    return main_args


if __name__ == '__main__':
    args = get_args()

    # Setup logger
    logging.basicConfig(format='%(asctime)s | %(name)s-%(levelname)s: %(message)s')
    logger = logging.getLogger(name="ErNESTO-DT")
    ch = logging.StreamHandler()

    if args['verbose']:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)

    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    # Parsing of models employed in the current experiment
    
    args['models'] = {}
    if args['electrical']:
        args['models']['electrical'] = args['electrical'][0]
        del args['electrical']

    if args['thermal']:
        args['models']['thermal'] = args['thermal'][0]
        del args['thermal']

    if args['aging']:
        args['models']['aging'] = args['aging'][0]
        del args['aging'] 
    
    # Parallel execution of the experiments
    parallel_exp_config = args['config_files']
    del args['config_files']

    n_cores = args['n_cores']
    del args['n_cores']
    
    if n_cores == 1:
        run_experiment(args, parallel_exp_config[0])
    else:
        Parallel(n_jobs=n_cores)(delayed(run_experiment)(args, config) for config in parallel_exp_config)

