import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
from src.digital_twin.simulator import Simulator


def get_args():
    parser = argparse.ArgumentParser(description="Digital Twin of a Battery Energy Storage System (RSE)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--data_folder",
                        action="store",
                        default="./src/data",
                        type=str,
                        help="Specifies the folder which we retrieve data from."
                        )
    parser.add_argument("--config",
                        action="store",
                        default="simulation_config.yaml",
                        type=str,
                        help="Specifies the file containing parameters useful for the simulation."
                        )

    models_choices = ['thevenin', 'thermal', 'bolun']
    parser.add_argument("--models",
                        choices=models_choices,
                        default=['thevenin'],
                        help="Specifies which models should be run during the simulation."
                        )

    input_args = vars(parser.parse_args())
    return input_args


if __name__ == '__main__':
    args = get_args()
    print(args)

    data_folder = args['data_folder']
    simulation_config = args['config']
    models = args['models']

    sim = Simulator(data_folder, simulation_config, models)
    sim.run()