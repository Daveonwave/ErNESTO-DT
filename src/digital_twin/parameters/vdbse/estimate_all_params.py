import numpy as np
from src.digital_twin.parameters.vdbse.loss_function import loss_function
from scipy.optimize import minimize


def estimate_all_params(i_batch, v_batch, dt, algorithm, number_restarts, lookup, scale_factors, verbose):
    """
    %   estimate_all_params estimate values of equvalent model parameters
    %
    %   INPUT:
    %       i_batch: array of Current measurements
    %       v_batch: array of Voltage measurements
    %       dt: sampling interval
    %       algorithm: algorithm used by fmincon
    %       number_restarts: number of restarts to avoid local minima
    %       lookup: lookup table
    %       scale_factors: array [Rs_scale Rp_scale C_scale] scale factors
    %       verbose: verbose integer (1->True, 0->False)
    %
    %   OUTPUT:
    %       Rs, Rp, C, SoC(tau), Qmax
    """

    assert np.size(i_batch) == np.size(v_batch)

    time = np.size(i_batch)
    init_time = 0
    input = np.zeros((np.size(i_batch),3))
    input[:,0] = v_batch
    input[:,1] = i_batch
    input[:,2] = -np.cumsum(i_batch)

    to_minimize = lambda t: loss_function(t, input, dt, lookup, scale_factors)

    # Define optimization options
    if verbose == 1:
        display_details = 'iter'
    else:
        display_details = 'off'

    minOptions = {
        'disp': display_details,
        'algorithm': algorithm,
        'gtol': 1e-10,
        'xtol': 1e-10,
        'maxiter': 1000,
        'maxfev': 1000
    }

    # Define bounds and random initialization
    lb = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e4])
    ub = np.array([1e2, 1e2, 1e2, 1, 1e6])

    np.random.seed(0)  # For reproducibility
    randmatrix = np.random.rand(number_restarts, len(ub))
    randmatrix = lb + randmatrix * (ub - lb)


    """
    THE IDEA IS THAT OF ONE SHOT OPTIMIZATION AND THEN IMPLEMENT THE MULTIPLE RESTART(I)!!!
    AND THE ERROR CORRECTION(II)
    """
    # Perform multiple restarts to avoid local minima
    results = []
    #for jj in range(number_restarts):
     #  try:
    results = minimize(to_minimize, randmatrix, bounds=list(zip(lb, ub)), options={'iprint': -1})
       # print("the optimization result is",results)
        #results.append(results)
      #except:
        #   continue

    x = results.x
    #print("this is the theta: ", x)
    # Select the best estimation among the restarts
    #best_result = min(results, key=lambda result: result.fun)
    #best_x = best_result.x

    # Correct the values according to scale factors
    rs = x[0]
    rp = x[1]
    c =  x[2]
    soc_tau = x[3]
    qmax = x[4]

    return x
