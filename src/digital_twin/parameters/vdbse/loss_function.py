import numpy as np
from src.digital_twin.parameters.vdbse.estimate_v import estimate_v

"""
module to define the loss_function
"""
def get_vocv(soc, lookup_table):
    """
    soc should be: soc = np.linspace[0,1, len(i_batch)]
    """
    # Extract the SoC and Vocv values from the lookup table
    lookup_soc = lookup_table[:, 0]  # Assuming SoC values are in the first column
    lookup_vocv = lookup_table[:, 1]  # Assuming Vocv values are in the second column

    # Perform linear interpolation
    vocv = np.interp(soc, lookup_soc, lookup_vocv, left=None, right=None, period=None)

    return vocv


def loss_function(x, u, dt, lookup, scale_factors):

    """
    %   loss_function estimate the value of the loss given state x
    %   and others parameters
    %
    %   INPUT:
    %       x: state vector to optimize [Rs, Rp, C, SoC(tau), Qmax]
    %       u: measurement vector
    %       dt: sampling interval
    %       lookup: lookup table
    %       scale_factors: scale factors vector for Rs, Rp and C
    %
    %   OUTPUT:
    %       l: loss value
    """
    soctovocv = lambda t: get_vocv(t,lookup)
    soc = x[3] + ( u[:,2] * (dt / x[4]))
    soc = np.concatenate((soc,lookup[:,1]))
    vocv = soctovocv(soc)

    v_est = estimate_v( u[1:,1], vocv[1:], u[0,1], u[0,0], vocv[0], dt, x[0],
                        x[1], x[2], scale_factors[0], scale_factors[1], scale_factors[2])

    v_est = v_est.reshape(-1, 1)  # Reshape to a column vector
    l = np.sum(np.abs(v_est[1:] - u[1:, 0]))

    return l