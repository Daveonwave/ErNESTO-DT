import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.digital_twin.estimators import SOCEstimator

def cc_soc(i_batch,init_soc,dt,q):
    soc_array = np.zeros(len(i_batch))
    t = init_soc
    for ii in range(len(i_batch)):
        t = t - (1 / q) * i_batch[ii] * dt
        soc_array[ii] = t
    return soc_array

lookup_data = pd.read_csv('../vdbseTest/lookup_fieldexperiments.csv')
data = pd.read_csv('../vdbseTest/data_fieldexperiments.csv')

lookup = lookup_data.to_numpy()

dt = 1
#time = 9599
timewindow = 0.4
moving_step = 0.2
number_restart = 1 #10
battery_capacity = 53.4 * 3600  # Convert to Coulombs
# battery_capacity = 50 * 3600
scale_factors = [0.1, 0.1, 10]

I = -data['I'].to_numpy()  # Assuming 'I' is a column in the CSV file
V = data['V'].to_numpy()
SoC = data['SoC'].to_numpy()

print("The I shape and the following V: ",I.shape,V.shape)

estimatorCC = SOCEstimator(I, V, lookup, battery_capacity, scale_factors, dt, moving_step, number_restart, timewindow, verbose = 1)
estimatorVDBSE = SOCEstimator(I, V, lookup, battery_capacity, scale_factors , 'VDBSE' , dt, moving_step, number_restart, timewindow, verbose = 1 )

for i,v in zip(I,V):
    soc_cc = estimatorCC.compute_soc()
    soc_vdbse = estimatorVDBSE.compute_soc()
time = len(soc_vdbse)

print("soc_vdbse is:", soc_vdbse)
print("its shape is:", np.shape(soc_vdbse))
print("soc_cc is:", soc_cc)
print("the time is", time)
#print("soc_cc shape is:" ,len(soc_cc))

#print("the shape of soc_vdbse[1:time-1] and soc_cc[1:time-1] ", soc_vdbse[1:time-1], soc_cc[1:time-1])
#SoC_mean_error = np.sum(np.abs(soc_vdbse[1:time-1] - soc_cc[1:time-1]))

# Plotting
interval = 100
fig, ax_soc = plt.subplots(figsize=(12, 6))
ax_soc.set_xlabel('t[s]')
ax_soc.set_ylabel('SoC(t)')
ax_soc.set_xlim([0, time * dt])
ax_soc.set_ylim([0, 1])
ax_soc.plot(np.arange(2, time, interval) * dt, soc_cc[2:time:interval], label='SoC', linewidth=1)
ax_soc.plot(np.arange(2, time, interval) * dt, soc_vdbse[2:time:interval], label='Estimated SoC', linewidth=1)
ax_soc.legend(loc='lower center')

fig_err, ax_err = plt.subplots(figsize=(12, 6))
ax_err.set_xlabel('t[s]')
ax_err.set_ylabel('SoC error')
ax_err.set_xlim([0, time * dt])
ax_err.set_ylim([0, 2000])
ax_err.plot(np.arange(2, time, interval) * dt, np.abs(soc_vdbse[2:time:interval] - soc_cc[2:time:interval]))
ax_err.legend(['Error'])

#err = np.mean(np.abs(soc_vdbse[1:time-1] - soc_cc[1:time-1]))
#print(err)

plt.show()