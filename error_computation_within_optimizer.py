import numpy as np
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.optimizer import Optimizer
from src.online_learning.utils import load_from_yaml
import pandas as pd

def percentage_error(true_value, approximate_value):
    error = abs(true_value - approximate_value)
    return (error / true_value) * 100


if __name__ == "__main__":
    # Hyperparameters:
    alpha = 0.00
    optimizer_method = 'BFGS'
    number_of_restarts = 1
    bounds = [(0.001, 0.5), (0.001, 0.5), (800, 20000)]
    scale_factor = None
    options = {
        'gtol': 1e-10,  # Gradient norm tolerance
        'eps': 1e-10,  # Step size for numerical gradient approximation
        'maxiter': 10000,  # Maximum number of iterations
        'disp': True  # Display convergence messages
    }  # BFGS options
    number_of_initial_guesses = 10

    # Load Dataframe
    df = pd.read_csv('data/ground/experiment_signals/dataset_0_cutted40000samples.csv')
    i_real = df['current'].values
    time = df['time']

    # Load YAML:
    grid_parameters = load_from_yaml('data/external/ranges')
    electrical_params = load_from_yaml('data/external/electrical_params')
    thermal_params = load_from_yaml('data/external/thermal_params')
    models_config = [electrical_params, thermal_params]
    battery_options = load_from_yaml('data/external/battery_options')
    load_var = 'current'

    battery = BatteryEnergyStorageSystem(
        models_config=models_config,
        battery_options=battery_options,
        input_var=load_var
    )

    reset_info = {'electricala_params': electrical_params, 'thermal_params': thermal_params}
    battery.reset(reset_info)
    battery.init({'dissipated_heat': 0})  # check if you can remove it
    elapsed_time = 0
    dt = 1
    v_optimizer = list()
    temp_optimizer = list()

    # battery_results = battery.get_last_results()
    optimizer = Optimizer(models_config=models_config, battery_options=battery_options, load_var=load_var,
                          init_info=reset_info, bounds=bounds, scale_factor=scale_factor, options=options)

    for k, load in enumerate(i_real):
        elapsed_time += dt
        battery.t_series.append(elapsed_time)
        dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0
        battery.step(load, dt, k)

    results = battery.build_results_table()
    # print(results['operations'])
    results = results['operations']
    v = results['voltage']
    v_real = np.array(v)
    v_real = v_real[1:len(v_real)]
    t = results['temperature']
    t_real = np.array(t)
    t_real = t_real[1:len(t_real)]

    # real theta
    r0 = battery._electrical_model.r0.resistance
    rc = battery._electrical_model.rc.resistance
    c = battery._electrical_model.rc.capacity
    theta_real = {'r0': r0, 'rc': rc, 'c': c}
    theta_real_values = np.array(list(theta_real.values()), dtype=float)

    list_of_theta_hat = list()
    for i in range(number_of_initial_guesses):
        theta_hat = optimizer.step(i_real=i_real, v_real=v_real,
                               t_real=t_real, optimizer_method=optimizer_method,
                               alpha=alpha, dt=dt, number_of_restarts=number_of_restarts,
                               starting_theta=np.array([np.random.uniform(low, high) for low, high in bounds]))
        # see also parallelization
        list_of_theta_hat.append(theta_hat)

    # todo: wrong in the following indeed wrong csv
    # todo: run again, since before you used theta_hat instead of theta 4hours
    list_of_theta_error = list()
    for theta in list_of_theta_hat:
        r0_error = percentage_error(true_value=theta_real['r0'], approximate_value=theta['r0'])
        rc_error = percentage_error(true_value=theta_real['rc'], approximate_value=theta['rc'])
        c_error = percentage_error(true_value=theta_real['c'],approximate_value=theta['c'])
        theta_error = {'r0_error': r0_error, 'rc_error': rc_error, 'c_error': c_error}
        list_of_theta_error.append(theta_error)

    r0_errors = []
    rc_errors = []
    c_errors = []
    for error in list_of_theta_error:
        r0_errors.append(error['r0_error'])
        rc_errors.append(error['rc_error'])
        c_errors.append(error['c_error'])

    r0_errors_array = np.array(r0_errors)
    r0_avg_percentage_error = np.mean(r0_errors_array)
    rc_errors_array = np.array(rc_errors)
    rc_avg_percentage_error = np.mean(rc_errors_array)
    c_errors_array = np.array(c_errors)
    c_avg_percentage_error = np.mean(c_errors_array)

    print(f"The r0 average percentage error is: {r0_avg_percentage_error:.2f}%")
    print(f"The rc average percentage error is: {rc_avg_percentage_error:.2f}%")
    print(f"The c average percentage error is: {c_avg_percentage_error:.2f}%")


    df_list_of_percentage_errors = pd.DataFrame(list_of_theta_error)
    df_list_of_percentage_errors.to_csv('percentage_errors', index=False)



