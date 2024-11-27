metrics_list = ['mse', 'mae', 'mape', 'max_abs_err']


def _mse(ground: list, simulated: list):
    """
    Compute the Mean Squared Error (MSE) between two lists of data.

    Args:
        ground (list): true values
        simulated (list): simulated values
    """    
    assert len(ground) == len(simulated), ("MSE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([(y - x)**2 for x, y in zip(simulated, ground)]) / len(simulated)


def _mae(ground: list, simulated: list):
    """
    Compute the Mean Absolute Error (MAE) between two lists of data.

    Args:
        ground (list): true values
        simulated (list): simulated values
    """
    assert len(ground) == len(simulated), ("MAE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([abs(y - x) for x, y in zip(simulated, ground)]) / len(simulated)


def _max_abs_err(ground: list, simulated: list):
    """
    Compute the Maximum Absolute Error between two lists of data.

    Args:
        ground (list): true values
        simulated (list): simulated values
    """
    assert len(ground) == len(simulated), ("MaAE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return max([abs(y - x) for x, y in zip(simulated, ground)])


def _mape(ground: list, simulated: list):
    """
    Compute the Mean Absolute Percentage Error (MAPE) between two lists of data.

    Args:
        ground (list): true values
        simulated (list): simulated values
    """
    assert len(ground) == len(simulated), ("MAPE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([abs(y - x) / y for x, y in zip(simulated, ground) if y != 0]) * 100 / len(simulated)


def compute_metrics(ground: dict, simulated: dict, vars: list, metrics:list=None, steps=None):
    """
    Method to compute metrics between ground truth and simulated data. 
    The metrics that can be computed are: Mean Squared Error (MSE), Mean Absolute Error (MAE),
    Maximum Absolute Error (MAX_ABS_ERR) and Mean Absolute Percentage Error (MAPE).

    Args:
        ground (dict): true data dictionary.
        simulated (dict): simulated data dictionary.
        vars (list): variables to compute metrics.
        metrics (list, optional): metrics to compute. Defaults to None.
        steps (int, optional): steps to consider in the computation. Defaults to None.
        
    Returns:
        dict: dictionary containing the computed metrics for each variable.
    """
    assert ground is not None, "Computation of metrics failed: ground data dictionary is None"
    assert simulated is not None, "Computation of metrics failed: simulation data dictionary is None"

    if metrics is None:
        metrics = metrics_list
    res_dict = {}

    for var in vars:
        res_dict[var] = {}
        for metric in metrics:
            try:
                ground_data = [ground[var][i] for i in steps] if steps is not None else ground[var]
                res_dict[var][metric] = globals()['_' + metric](ground_data, simulated[var])
            except KeyError:
                print("It wasn't possible to compute metric '{}' for variable '{}'. "
                      "The possible metrics to compute are {}.".format(metric, var, metrics_list))
                res_dict[var][metric] = None

    return res_dict
