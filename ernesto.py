import argparse
import logging
import sys
from joblib import Parallel, delayed
from src.utils.logger import CustomFormatter
from src.digital_twin.orchestrator.orchestrator import DTOrchestrator


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
        main_parser.add_argument("--config_folder", action="store", default="./data/config", type=str,
                                 help="Specifies the folder which we retrieve preprocessing from.")

        main_parser.add_argument("--output_folder", action="store", default="./data/output", type=str,
                                 help="Specifies the name of the folder where to store the output results.")

        main_parser.add_argument("--ground_folder", action="store", default="./data/ground", type=str,
                                 help="Specifies the folder which we retrieve preprocessing from.")

        main_parser.add_argument("--assets", action="store", default="./data/config/assets.yaml",
                                 type=str, help="Specifies the file containing parameters useful for the experiment.")

        electrical_choices = ['first_order_thevenin', 'second_order_thevenin']
        main_parser.add_argument("--battery_model", nargs=1, choices=electrical_choices, default=['first_order_thevenin'],
                                 help="Specifies the name of the core model of the battery, electrical or data driven.")

        thermal_choices = ['rc_thermal', 'r2c_thermal', 'dummy_thermal', 'mlp_thermal']
        main_parser.add_argument("--thermal_model", nargs=1, choices=thermal_choices, default=['r2c_thermal'],
                                 help="Specifies the name of the thermal model that has to be used.")

        aging_choices = ['bolun']
        main_parser.add_argument("--aging_model", nargs=1, choices=aging_choices,
                                 help="Specifies the name of the aging model that has to be used.")

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
    args['models'] = []
    if args['battery_model']:
        args['models'].extend(args['battery_model'])
        del args['battery_model']

    if args['thermal_model']:
        args['models'].extend(args['thermal_model'])
        del args['thermal_model']

    if args['aging_model']:
        args['models'].extend(args['aging_model'])
        del args['aging_model']

    parallel_exp_config = args['config_files']
    del args['config_files']

    n_cores = args['n_cores']
    del args['n_cores']
    
    if n_cores == 1:
        run_experiment(args, parallel_exp_config[0])
    else:
        Parallel(n_jobs=n_cores)(delayed(run_experiment)(args, config) for config in parallel_exp_config)

