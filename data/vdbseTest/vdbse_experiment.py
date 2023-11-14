import  pandas as pd
import numpy as np
from scipy.special import expit
from src.digital_twin.estimators import SOCEstimator

"""
Notice that the data used right now in this file are not the same as the data of the matlab experiment.
"""

def apply_biases(i_input, v_input, i_offset, i_gain, v_offset, v_gain, lookup_input, apply_to_vocvsoc):
    i_tmp = i_input * i_gain + i_offset
    v_tmp = v_input * v_gain + v_offset

    if apply_to_vocvsoc == 1:
     lookup_input[:,1] = lookup_input[:,1] * v_gain + v_offset

    return [i_tmp, v_tmp, lookup]

def cc_soc(i_batch,init_soc,dt,q):

    soc_array = np.zeros(len(i_batch))  # Create an array to store SoC values

    t = init_soc
    for ii in range(len(i_batch)):
        t = t - (1 / q) * i_batch[ii] * dt
        soc_array[ii] = t

    return soc_array

"""
the experiment starts here:
"""

path1 = '../vdbseTest/power_profile/01_powerprofile_10degrees_0capacityreduction.csv'
path2 = '../vdbseTest/power_profile/01_powerprofile_10degrees_0capacityreductionintegers.csv'
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
tmp1 = df1.to_numpy()
v = tmp1[:200,6]
i = tmp1[:200,1]

tmp2 = df2.to_numpy()
battery_capacity = tmp2[0,0]
dt = tmp2[0,2]

print("The dimensions of the two time series used by the algorithm : \n")
print("len of i : ", len(i))
print("len of v : ", len(v))

error_list = [[0, 1, 0, 1, 0], [0.05, 1.01, 0.005, 1.001, 0], [0.05, 1.01, 0.005, 1.001, 1]]
errors = np.array(error_list)

#the following depends on the specific battery and it is given
f_characteristic = lambda x: 3.43 + 0.68 * x - 0.68 * (x ** 2) + 0.81 * (x ** 3) - 0.31 * expit(-46 * x)

#lookup is intended to be as numpy matrix of dim x 2
soc = np.linspace(0,1,100)
vocv = np.array([f_characteristic(x) for x in soc])

lookup = np.zeros((len(i),2))
lookup[:,0] = np.array(i)
lookup[:,1] = np.array(v)

soc_vdbse = np.zeros(len(i))

for k in i:

    for err_i in range(2):
       [i,v,lookup] = apply_biases(i, v, errors[err_i,0], errors[err_i,1], errors[err_i,2], errors[err_i,3], errors[err_i,4], 0)
       """
       WARNING TAKE CARE THAT THE NUMERIC VALUE ARE ASSUMED ONLY TO TEST THE CODE
       """
       soc_cc = cc_soc(i,0.55, dt , battery_capacity)
       #print(' & {:.4f}'.format(np.mean(np.abs(soc_cc[0:len(i)] - soc[0:len(i)]))))

       #da togliere!!!!!
       experiment = SOCEstimator(i, v, lookup, battery_capacity, [1e-3,1e-3,1e3], 'VDBSE',dt, 0.4
                                 , 1, 0.2, 1)
       soc_vdbse = experiment.compute_soc()

time = len(i)
err_soc_vdbse = abs(soc_vdbse[-100:] - soc[0:100])
err_soc_cc = abs(soc_cc[-100:] - soc[0:100])

print("the err_soc_vdbse: ", err_soc_vdbse)
print("the err_soc_cc: ", err_soc_cc)






