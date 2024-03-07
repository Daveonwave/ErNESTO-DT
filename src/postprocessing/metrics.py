import logging

logger = logging.getLogger('DT_ernesto')
metrics_list = ['mse', 'mae', 'mape', 'max_abs_err']


def _mse(ground: list, simulated: list):
    """
    """
    assert len(ground) == len(simulated), ("MSE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([(y - x)**2 for x, y in zip(simulated, ground)]) / len(simulated)


def _mae(ground: list, simulated: list):
    """
    """
    assert len(ground) == len(simulated), ("MAE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([abs(y - x) for x, y in zip(simulated, ground)]) / len(simulated)


def _max_abs_err(ground: list, simulated: list):
    """
    """
    assert len(ground) == len(simulated), ("MAE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return max([abs(y - x) for x, y in zip(simulated, ground)])


def _mape(ground: list, simulated: list):
    """
    """
    assert len(ground) == len(simulated), ("MAPE: ground and simulated data have different lengths ({}, {})".
                                           format(len(ground), len(simulated)))
    return sum([abs(y - x) / y for x, y in zip(simulated, ground)]) * 100 / len(simulated)


def compute_metrics(ground: dict, simulated: dict, vars: list, metrics=None, steps=None):
    """

    Args:
        ground ():
        simulated ():
        vars ():
        metrics ():
        steps (): consider a subset of iterations of the ground data if the length differ from simulated ones
    """
    assert ground is not None, logger.error("Computation of metrics failed: ground data dictionary is None")
    assert simulated is not None, logger.error("Computation of metrics failed: simulation data dictionary is None")

    if metrics is None:
        metrics = metrics_list
    res_dict = {}

    for var in vars:
        res_dict[var] = {}
        for metric in metrics:
            try:
                ground_data = [ground[var][i] for i in steps] if steps is not None else ground[var]
                res_dict[var][metric] = globals()['_' + metric](ground_data, simulated[var][1:])
            except KeyError:
                logger.error("It wasn't possible to compute metric '{}' for variable '{}'. "
                             "The possible metrics to compute are {}.".format(metric, var, metrics_list))
                res_dict[var][metric] = None

    return res_dict
