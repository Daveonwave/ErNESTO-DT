import argparse
import os
from pathlib import Path
from src.digital_twin.dt_manager import DTManager


def get_args():
    parser = argparse.ArgumentParser(description="Digital Twin of a Battery Energy Storage System (RSE)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d", "--data-folder",
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
                        default=['thevenin', 'rc_thermal', 'bolun'],
                        help="Specifies which electrical should be run during the experiment."
                        )

    mode_choices = ['simulation', 'what-if', 'learning', 'optimization']
    parser.add_argument("-m", "--mode",
                        choices=mode_choices,
                        default=['simulation'],
                        help="Specifies the working mode of the Digital Twin.")

    parser.add_argument("-s", "--save-results",
                        action="store_true",
                        default=False,
                        help="Specifies if it is necessary to save computed data at the end of the experiment."
                        )

    parser.add_argument("--plot",
                        action="store_true",
                        default=False,
                        help="Specifies if it is necessary to immediately plot computed data at the end of the "
                             "experiment."
                        )

    input_args = vars(parser.parse_args())
    return input_args


if __name__ == '__main__':
    args = get_args()

    data_folder = args['data_folder']
    simulation_config = args['configs']
    models = args['models']
    mode = args['mode']
    save_flag = args['save_results']
    plot_flag = args['plot']

    data_folders_dict = dict(
        config_data="configuration",
        load_data="load",
        output_data="output",
        ground_data="ground"
    )

    data_folder_paths = {}

    for key in data_folders_dict.keys():
        if not os.path.exists(data_folder + '/' + data_folders_dict[key]):
            raise NotADirectoryError("Folder '{}' is not a present inside 'data' folder. "
                                     "Folder 'data' has to contain the following sub-folders to run the simulation:\n"
                                     "\t- 'configuration': contains the configuration file which can be selected with"
                                     " --config argument;\n"
                                     "\t- 'load': contains the input data;\n"
                                     "\t- 'output': will contains experiment outputs;\n"
                                     "\t- 'ground': contains real world data.\n"
                                     .format(data_folders_dict[key]))
        else:
            data_folder_paths[key] = Path(data_folder + '/' + data_folders_dict[key])

    sim = DTManager(experiment_mode=mode[0],
                    experiment_config=simulation_config,
                    models=models,
                    save_results=save_flag,
                    plot_results=plot_flag,
                    **data_folder_paths
                    )
    sim.run()