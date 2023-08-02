import time
import yaml
import logging
import os
import pint.util
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from pint import UnitRegistry

ureg = UnitRegistry()
logger = logging.getLogger('DT_logger')


# Dictionary of units internally used inside the simulator
internal_units = dict(
    current=['ampere', 'A', ureg.ampere],
    voltage=['volt', 'V', ureg.volt],
    power=['watt', 'W', ureg.watt],
    resistance=['ohm', '\u03A9', ureg.ohm],
    capacity=['faraday', 'F', ureg.faraday],
    temperature=['celsius', 'degC', ureg.degC],
    time=['seconds', 's', ureg.s]
)


def load_data_from_csv(csv_file: Path, vars_to_retrieve: [dict], **kwargs):
    """
    Function to preprocess data that need to be read from a csv table.

    Inputs:
    :param csv_file: file path of the csv which we want to retrieve data from
    :param vars_to_retrieve: variables to retrieve from csv file
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))

    df = None
    try:
        df = pd.read_csv(csv_file, encoding='unicode_escape')
        if kwargs['iterations']:
            df = df.iloc[:kwargs['iterations']]
    except IOError:
        logger.error("The specified file '{}' cannot be imported as a Pandas Dataframe.".format(csv_file))

    # Retrieve and convert timestamps to list of seconds (format: YYYY/MM/DD hh:mm:ss)
    timestamps = pd.to_datetime(df['Time'], format="%Y/%m/%d %H:%M:%S").values.astype(float) // 10 ** 9
    vars_data = {}

    # We first check if the variable column label exists
    for var in vars_to_retrieve:
        if var['label'] not in df.columns:
            raise NameError("Label {} is not present among df columns [{}]".format(var['label'], df.columns))
        else:
            vars_data[var['var']] = _validate_data_unit(df[var['label']].values.tolist(), var['var'], var['unit'])

    return vars_data, timestamps.tolist()


def _validate_data_unit(data_list, var_name, unit):
    """
    Function to validate and adapt data unit to internal simulator units.

    Inputs:
    :param data_list: list with values of a data stream
    :param var_name: name of the variable
    :param unit: unit of the variable
    """
    # Unit employed is already compliant with internal simulator units
    if unit == internal_units[var_name][1]:
        return data_list

    try:
        tmp_data = data_list * ureg.parse_units(unit)
        transformed_data = tmp_data.to(internal_units[var_name][2])
        logger.info("Ground variable '{}' has been converted from [{}] to [{}]"
                    .format(var_name, unit, internal_units[var_name][1]))
    except pint.PintError as e:
        logger.error("UnitError on '{}': ".format(var_name), e)
        exit(1)

    return transformed_data.magnitude.tolist()


def validate_parameters_unit(param_dict):
    """
    Function to validate and adapt units of provided parameters to internal simulator units.

    Inputs:
    :param param_dict: dictionary of parameters (read by for example yaml configuration file)
    """
    transformed_dict = {}

    for key in param_dict.keys():
        param = param_dict[key]

        # Check if the parameter has a unit measure with a dictionary structure
        if type(param) == dict:
            # Parameter unit measure is not compliant with internal simulator units
            if param['unit'] != internal_units[param['var']][1]:
                try:
                    tmp_param = param['value'] * ureg.parse_units(param['unit'])
                    transformed_dict[key] = tmp_param.to(internal_units[param['var']][2]).magnitude
                except pint.PintError as e:
                    logger.error("UnitError on '{}': ".format(param['var']), e)
                    exit(1)
            else:
                transformed_dict[key] = param['value']
        else:
            transformed_dict[key] = param_dict[key]

    return transformed_dict
