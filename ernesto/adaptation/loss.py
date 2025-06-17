import numpy as np
from scipy.stats import skew, kurtosis
from ernesto.digital_twin.parameters.variables import instantiate_variables
from ernesto.digital_twin.battery_models.electrical.ecm_components import OCVGenerator

def gaussian_penalty(data: np.ndarray):
    sk = np.abs(skew(data))        # Skewness ideally should be 0
    kt = (kurtosis(data) - 3)**2   # Kurtosis ideally should be 3 (excess kurtosis 0)
    return sk + kt


def loss_first_order_thevenin(params: list, 
                              input_batch: list,
                              init_state: dict, 
                              battery_config: dict, 
                              alpha: float,
                              beta: float):
    
    # Read the parameters of electrical model
    r0, r1, c = params
    v_rc = 0   
    c_max = init_state['c_max']
    v = init_state['voltage']
    q = 0    
    
    # Parse the components that change within the model
    components = {}
    
    for model in battery_config['models_config']:
        if model['type'] in 'electrical':
            components['v_ocv'] = model['components']['v_ocv']
        if model['type'] in 'thermal':
            components['dVoc_dT'] = model['components']['dVoc_dT']
            c_term = model['components']['c_term']['scalar']
            r_cond = model['components']['r_cond']['scalar']
            r_conv = model['components']['r_conv']['scalar']
    
    # Instantiate the OCV generator
    init_components = instantiate_variables(components)
    ocv_gen = OCVGenerator(name='ocv', ocv_potential=init_components['v_ocv'])
    ocv_gen.reset_data()
    dVoc_dT_table = init_components['dVoc_dT']
    
    # SoC and temperature
    soc = init_state['soc']
    temp = init_state['temperature']
    ocv_gen.soc = soc
    ocv_gen.temp = temp
    
    estimated_v = []
    for k, sample in enumerate(input_batch[:-1]):
        # Normal operating step of the battery system.
        t_amb = sample['t_amb'] if 't_amb' in sample else init_state['t_amb']
        dt = input_batch[k+1]['time'] - sample['time']
        
        # Electrical model operation
        v_ocv = ocv_gen.ocv_potential
        
        if battery_config['battery_options']['sign_convention'] == 'passive':
            i = -sample['current']
        
        # Compute V_r0 and V_rc
        v_r0 = r0 * i
        v_rc = (v_rc/dt + i/c) / (1/dt + 1 / (c * r1))
        i_rc = v_rc / r1

        # Compute V
        v = v_ocv - v_r0 - v_rc
        estimated_v.append(v)
        
        # If convention was 'passive', then we need to change the sign of the current
        i = sample['current']
        
        # Compute SoC with Coulomb counting
        soc = np.clip(soc + i / (c_max * 3600) * dt, a_min=0, a_max=1)
        
        # Compute temperature with R2C thermal model
        q = r0 * i**2 + r1 * i_rc**2
        dVoc_dT = dVoc_dT_table.get_value(input_vars={'soc': soc})
        
        term_1 = c_term / dt * temp
        term_2 = t_amb / (r_cond + r_conv)
        denominator = c_term/dt + 1/(r_cond + r_conv) - (dVoc_dT * i)

        t_core = (term_1 + term_2 + q) / denominator
        temp = t_core + r_cond * (t_amb - t_core) / (r_cond + r_conv)
        
        # Update the OCV generator
        ocv_gen.soc = soc
        ocv_gen.temp = temp
    
    # Compare the estimated voltage with the real one with the MSE
    estimated_v = np.array(estimated_v)
    true_voltage = np.array([sample['voltage'] for sample in input_batch[1:]])
    voltage_loss = np.sum((estimated_v - true_voltage) ** 2)
    """
    temperature_loss = 0
    if self.temperature_loss:
        self.t_hat = self._dual_battery._thermal_model.get_temp_series()
        self.t_hat = self.t_hat[1: len(self._t_real) + 1]  # Adjusting for alignment
        temperature_diff = self.t_hat - self._t_real
        temperature_loss = np.sum(temperature_diff ** 2)
    """
    #print(f"Voltage loss: {voltage_loss}")
    regularization = alpha * np.sum(np.array(params) ** 2)
    reg_gauss = gaussian_penalty(estimated_v)
    #print("Total loss: ", voltage_loss + regularization, "regularization: ", regularization)
    return voltage_loss + regularization + beta * reg_gauss


def scaled_loss(scaled_params, input_batch, init_state, battery_config, scale_factors, alpha, beta):
    # Unscale parameters
    params = np.array(scaled_params) / scale_factors
    return loss_first_order_thevenin(params, input_batch, init_state, battery_config, alpha, beta)