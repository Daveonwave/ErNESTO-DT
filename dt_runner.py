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
                        default="./data",
                        type=str,
                        help="Specifies the folder which we retrieve data from."
                        )

    parser.add_argument("--configs",
                        action="store",
                        default="experiment_config.yaml",
                        type=str,
                        help="Specifies the file containing parameters useful for the experiment."
                        )

    models_choices = ['thevenin', 'rc_thermal', 'r2c_thermal', 'bolun']
    parser.add_argument("--models",
                        nargs='*',
                        choices=models_choices,
                        default=['thevenin'],
                        help="Specifies which electrical should be run during the experiment."
                        )

    mode_choices = ['simulation', 'what-if', 'learning', 'optimization']
    parser.add_argument("-m", "--mode",
                        choices=mode_choices,
                        default=['simulation'],
                        help="Specifies the working mode of the Digital Twin.")

    input_args = vars(parser.parse_args())
    return input_args


if __name__ == '__main__':
    args = get_args()

    data_folder = args['data_folder']
    simulation_config = args['configs']
    models = args['models']
    mode= args['mode']

    data_folders_dict = dict(
        config_data = 'configuration',
        load_data = 'load',
        output_data = 'output',
        ground_data = 'ground'
    )

    data_folder_paths = {}

    for key in data_folders_dict.keys():
        if not os.path.exists(data_folder + '/' + data_folders_dict[key]):
            raise NotADirectoryError("Folder '{}' is not a present inside 'data' folder. "
                                     "Folder 'data' has to contain the following sub-folders to run the simulation:\n"
                                     "\t- 'configuration'\n"
                                     "\t- 'load'\n"
                                     "\t- 'output'\n"
                                     "\t- 'ground'\n"
                                     .format(data_folders_dict[key]))
        else:
            data_folder_paths[key] = Path(data_folder + '/' + data_folders_dict[key])

    sim = Simulator(mode=mode[0],
                    simulation_config=simulation_config,
                    models=models,
                    **data_folder_paths
                    )
    sim.run()