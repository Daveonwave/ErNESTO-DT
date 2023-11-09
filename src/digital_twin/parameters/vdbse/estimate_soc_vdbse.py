import numpy as np
from src.digital_twin.parameters.vdbse.estimate_all_params import estimate_all_params
from src.digital_twin.parameters.vdbse.estimate_vocv import estimate_vocv

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

def estimate_soc_vdbse(i, v, battery_capacity, dt, soctimewindow, moving_step, restarts, lookup, scale_factors,
                       verbose):
    """
    %   estimate_soc_VDBSE estimates the SoC value SoC(t) for each t
%
%   INPUT:
%       I: array of Current measurements
%       V: array of Voltage measurements
%       battery_capacity: nominal battery capacity
%               (not used in the estimation)
%       dt: sampling interval
%       SoCtimewindow: amplitude of the SoC variation required to
%               select the data batch for estimation
%       moving_step: SoC variation which trigger the new model
%               estimation
%       restarts: number of restarts to avoid local minima
%       lookup: lookup table
%       scale_factors: array [Rs_scale Rp_scale C_scale] scale factors
%       time: total time
%       verbose: verbose integer (1->True, 0->False)
%
%   OUTPUT:
%       SoC(t)
    """

    iter_init = 0
    iter_end = 0
    actual_SoC = 0
    min_SoC = 0
    max_SoC = 0
    time = np.size(i)

    while max_SoC - min_SoC < soctimewindow and iter_end < time-1:
        actual_SoC = actual_SoC - (1 / battery_capacity) * i[iter_end] * dt
        iter_end = iter_end + 1
        max_SoC = max(max_SoC, actual_SoC)
        min_SoC = min(min_SoC, actual_SoC)

    windows_amplitude = iter_end - iter_init

    theta = np.zeros(5)

    # First estimation phase
    theta = estimate_all_params( i[iter_init:iter_end+1], v[iter_init:iter_end+1], dt, 'interior point',
                                 restarts, lookup, scale_factors, verbose )


    # First prediction phase
    vocv_est = estimate_vocv(i[iter_init:iter_end+1], v[iter_init:iter_end+1], i[iter_init], v[iter_init], get_vocv(theta[3],lookup)
                             ,dt , theta[0], theta[1], theta[2], 1, 1, 1)


    while iter_end < time :

        iter_init = iter_end + 1
        min_SoC = 0
        max_SoC = 0
        iter_end = iter_init
        actual_SoC = 0

        #vocv_est_size = len(vocv_est)
        #if iter_end >= vocv_est_size:
        #    iter_end = vocv_est_size - 1

        while max_SoC - min_SoC < moving_step and iter_end < time - 1:
            actual_SoC = actual_SoC - (1 / battery_capacity) * i[iter_end] * dt
            iter_end = iter_end + 1
            max_SoC = max(max_SoC, actual_SoC)
            min_SoC = min(min_SoC, actual_SoC)

            # Vocv is estimated using the old model until the new one is available
            print("the dimensions to check are: ", vocv_est[iter_init:iter_end+1], i[iter_init:iter_end+1],v[iter_init:iter_end+1], i[iter_init], v[iter_init] )
            vocv_est[iter_init:iter_end+1] = estimate_vocv(i[iter_init:iter_end+1], v[iter_init:iter_end+1], i[iter_init], v[iter_init], get_vocv(theta[3],lookup)
                             ,dt , theta[0], theta[1], theta[2], 1, 1, 1)

            #a new model is now estimated, it will be used in the next iteration of the for loop

            theta = estimate_all_params(i[iter_end-windows_amplitude:iter_end+1],v[iter_end-windows_amplitude:iter_end+1],
                                dt, 'interior-point', restarts, lookup, scale_factors, verbose )

        iter_end = iter_end + 1

        # estimate the value of the SoC(t) starting from the Vocv(t)
    lookup_soc = lookup[:, 0]  # Assuming SoC values are in the first column
    lookup_vocv = lookup[:, 1]
    soc = np.interp(vocv_est[0:time+1], lookup_vocv, lookup_soc)

    return soc