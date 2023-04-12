from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import os


def retrieve_data_from_csv(csv_file:Path, var_label:str, **kwargs):
    """
    # TODO: Add kwargs options
    """
    # Check file existence
    if not os.path.isfile(csv_file):
        raise FileNotFoundError("The specified file '{}' doesn't not exist.".format(csv_file))

    try:
        df = pd.read_csv(csv_file, encoding = 'unicode_escape')
    except:
        raise IOError("The specified file '{}' cannot be imported as a Pandas Dataframe.".format(csv_file))

    # Labels that have to be retrieved -> add kwargs labels
    labels_to_get = [var_label, 'Time']
    var_data = []
    timestamps = []
    others = {}

    for label in labels_to_get:
        # We first check if the label exists,
        if not label in df.columns:
            raise NameError("Label {} is not present between csv file columns names [{}]".format(label, df.columns))

        # Convert the time column to list of seconds (format: YYYY/MM/DD hh:mm:ss)
        if label == 'Time':
            timestamps = pd.to_datetime(df['Time'], format="%Y/%m/%d %H:%M:%S").values.astype(float) // 10 ** 9

        elif label == var_label:
            var_data = df[label].values.tolist()

        else:
            others[label] = df[label].values.tolist()

    return var_data, timestamps.tolist(), others

