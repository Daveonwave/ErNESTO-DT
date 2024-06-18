import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from src.online_learning import cluster as cl
from src.online_learning import mountain_method as mm
from src.online_learning import minimum_covariance_determinant as mcd
import numpy as np


def estimate_cdf(history_theta):
    """
    Estimate the empirical CDF for univariate data extracted from dictionaries.

    Args:
    history_theta (list of dict): A list of dictionaries, each containing scalars under keys like 'r0', 'rc', and 'c'.

    Returns:
    tuple: Sorted data and the corresponding empirical CDF values.
    """
    data = []

    for item in history_theta:
        for key in ['r0', 'rc', 'c']:
            if key in item:
                value = item[key]
                if isinstance(value, (int, float)):
                    data.append(value)
                else:
                    raise ValueError(f"The value for {key} must be a scalar (int or float).")

    if not data:
        raise ValueError("The input list is empty or contains no valid data.")

    # Convert list of scalars to a numpy array
    data_array = np.array(data)
    # Sort the data array
    data_sorted = np.sort(data_array)
    # Calculate the CDF values
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    return data_sorted, cdf


def induced_cdf(cluster_parameters):
    # Flatten the list of arrays into a single list of numbers


    flattened_data = np.concatenate(cluster_parameters)

    # Sort the flattened data
    sorted_data = np.sort(flattened_data)

    # Calculate CDF values
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    return sorted_data, cdf_values


def fault_cluster_creation(cluster_parameters, outliers_set):
    if len(outliers_set) < 20:
        return None

    x_hat, f_hat = estimate_cdf(outliers_set)
    print(f"x_hat: {x_hat}")
    print(f"x_hat length: {len(x_hat)}")
    if len(x_hat) == 0:
        print("Error: No data in x_hat")
        return None

    x_tau, f_tau = induced_cdf(cluster_parameters)
    print(f"x_tau: {x_tau}")
    print(f"x_tau length: {len(x_tau)}")
    if len(x_tau) == 0:
        print("Error: No data in x_tau")
        return None

    print(x_hat == x_tau)

    _, p_value = stats.ks_2samp(x_hat, x_tau)
    alpha_c = 0.05

    if p_value < alpha_c:
        print("Reject the null hypothesis: f_hat and f_tau are different. Create a new cluster.")
        mc = mm.MountainCluster()
        print(outliers_set)

        r0 = []
        rc = []
        c = []
        for th in outliers_set:
            r0.append(th['r0'])
            rc.append(th['rc'])
            c.append(th['c'])
        r0 = np.array(r0)
        rc = np.array(rc)
        c = np.array(c)
        array = np.column_stack([r0, rc, c])

        print("preprocess on outliers set")
        print(array)

        # TODO: UNDERSTAND HOW TO SEYT THE TRASHOLD
        O_tilde = mc.mountain_method(data=array, threshold=0.000000000000000)
        phi = mcd.create_cluster(O_tilde, p=20, support_fraction=0.75)
        return phi
    else:
        print("Fail to reject the null hypothesis: f_hat and f_tau are the same. No new cluster needed.")
        return None

