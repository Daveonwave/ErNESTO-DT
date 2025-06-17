import numpy as np
cimport numpy as np
from libc.math cimport pow

# Declare types for numpy
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def scaled_loss(np.ndarray[DTYPE_t, ndim=1] scaled_params,
                 list input_batch,
                 dict init_state,
                 dict battery_config,
                 np.ndarray[DTYPE_t, ndim=1] scale_factors,
                 float alpha,
                 float beta):

    cdef np.ndarray[DTYPE_t, ndim=1] params = scaled_params / scale_factors
    return loss_first_order_thevenin(params, input_batch, init_state, battery_config, alpha, beta)

def loss_first_order_thevenin(np.ndarray[DTYPE_t, ndim=1] params,
                               list input_batch,
                               dict init_state,
                               dict battery_config,
                               float alpha,
                               float beta):

    cdef double r0, r1, c, v_rc, c_max, v, q
    cdef double soc, temp, t_amb, dt, v_ocv, i, v_r0, i_rc
    cdef double term_1, term_2, denominator, t_core
    cdef int k, n = len(input_batch)
    cdef list estimated_v = []

    r0 = params[0]
    r1 = params[1]
    c  = params[2]

    v_rc = 0.0
    c_max = init_state['c_max']
    v = init_state['voltage']
    q = 0.0

    soc = init_state['soc']
    temp = init_state['temperature']
    t_amb = init_state['t_amb'] if 't_amb' in init_state else 25.0

    # Parse model components (left in Python domain)
    components = {}
    for model in battery_config['models_config']:
        if model['type'] in 'electrical':
            components['v_ocv'] = model['components']['v_ocv']
        if model['type'] in 'thermal':
            components['dVoc_dT'] = model['components']['dVoc_dT']
            c_term = model['components']['c_term']['scalar']
            r_cond = model['components']['r_cond']['scalar']
            r_conv = model['components']['r_conv']['scalar']

    # Using Python OCV generator (not optimized in Cython)
    from ernesto.digital_twin.parameters.variables import instantiate_variables
    from ernesto.digital_twin.battery_models.electrical.ecm_components import OCVGenerator
    init_components = instantiate_variables(components)
    ocv_gen = OCVGenerator(name='ocv', ocv_potential=init_components['v_ocv'])
    ocv_gen.reset_data()
    dVoc_dT_table = init_components['dVoc_dT']

    ocv_gen.soc = soc
    ocv_gen.temp = temp

    for k in range(n - 1):
        sample = input_batch[k]
        next_sample = input_batch[k+1]

        t_amb = sample['t_amb'] if 't_amb' in sample else t_amb
        dt = next_sample['time'] - sample['time']

        v_ocv = ocv_gen.ocv_potential

        if battery_config['battery_options']['sign_convention'] == 'passive':
            i = -sample['current']

        v_r0 = r0 * i
        v_rc = (v_rc/dt + i/c) / (1/dt + 1 / (c * r1))
        i_rc = v_rc / r1

        v = v_ocv - v_r0 - v_rc
        estimated_v.append(v)

        i = sample['current']
        soc += i / (c_max * 3600) * dt
        soc = max(0.0, min(1.0, soc))

        q = r0 * i**2 + r1 * i_rc**2
        dVoc_dT = dVoc_dT_table.get_value(input_vars={'soc': soc})

        term_1 = c_term / dt * temp
        term_2 = t_amb / (r_cond + r_conv)
        denominator = c_term / dt + 1 / (r_cond + r_conv) - (dVoc_dT * i)

        t_core = (term_1 + term_2 + q) / denominator
        temp = t_core + r_cond * (t_amb - t_core) / (r_cond + r_conv)

        ocv_gen.soc = soc
        ocv_gen.temp = temp

    cdef double loss = 0.0
    cdef double diff
    for k in range(len(estimated_v)):
        diff = estimated_v[k] - input_batch[k + 1]['voltage']
        loss += diff * diff

    cdef double regularization = alpha * (params[0]**2 + params[1]**2 + params[2]**2)
    return loss + regularization
