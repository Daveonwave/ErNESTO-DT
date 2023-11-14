import numpy as np

def estimate_v(i_batch, vocv_batch, last_i, last_v, last_vocv, dt, rs, rp, c, rs_scale, rp_scale, c_scale):
    """
    %   estimate_V estimates V given parameters and Current
        %   and Voltage measurements
        %
        %   INPUT:
        %       i_batch: I(t)
        %       vocv_batch: Vocv(t)
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
    #assert len(i_batch) == len(vocv_batch)

    c = c * c_scale
    rp = rp * rp_scale
    rs = rs * rs_scale

    v_batch = np.zeros(np.shape(i_batch))
    # assuming to work with 1D-array
    v_batch = np.concatenate((np.atleast_1d(last_v), v_batch))
    vocv_batch = np.concatenate((np.atleast_1d(last_vocv), vocv_batch))
    i_batch = np.concatenate((np.atleast_1d(last_i), i_batch))

    for j in range(len(i_batch)):
        v_batch[j] = ((1 / dt) * v_batch[j - 1] + ((1 / dt) + 1 / (c * rp)) * vocv_batch[j] - (1 / dt) * vocv_batch[
            j - 1] - ((rs / dt) + (1 / c) * (rs / (rp * c))) * i_batch[j] + (rs / dt) * i_batch[j - 1]) / (
                                 (1 / dt) + 1 / (c * rp))

    return v_batch[1:]

