import numpy as np
"""
This is the module to compute the vocv^, that is placed in the v estimation
"""

def estimate_vocv(i_batch, v_batch, last_i, last_v, last_vocv, dt, rs, rp, c, rs_scale, rp_scale, c_scale):
    """
    %   estimate_Vocv estimates Vocv given previously detected
    %   parameters and Current and Voltage measurements
    %
    %   INPUT:
    %       i_batch: I(t)
    %       v_batch: V(t)
    %       last_i: Current measurement before the data batch under
    %       analisys
    %       last_v: Voltage measurement before the data batch under
    %       analisys
    %       last_vocv: last value of Vocv estimated
    %       dt: sampling interval
    %       Rs: Rs value
    %       Rp: Rp value
    %       C: C value
    %       Rs_scale: Rs scale value
    %       Rp_scale: Rp scale value
    %       C_Scale: C scale value
    %
    %   OUTPUT:
    %       vocv_batch_returned: Vocv(t)
    """

    c = c * c_scale
    rp = rp * rp_scale
    rs = rs * rs_scale

    vocv_batch = np.zeros(np.shape(i_batch))
    # assuming to work with 1D-array
    vocv_batch = np.concatenate((np.atleast_1d(last_vocv), vocv_batch))
    v_batch = np.concatenate((np.atleast_1d(last_v), v_batch))
    i_batch = np.concatenate((np.atleast_1d(last_i), i_batch))

    for j in range(len(i_batch)):
        vocv_batch[j] = (((1 / dt) + (1 / (c * rp))) * v_batch[j] - (v_batch[j - 1] / dt) + (vocv_batch[j - 1] / dt)
                         + (((rs / dt) + (1 / c) + (rs / (c * rp))) * i_batch[j]) - ((rs / dt) * i_batch[j - 1])) / (
                                    (1 / dt) + (1 / (c * rp)))

    return vocv_batch[1:]