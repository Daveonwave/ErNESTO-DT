import pandas as pd
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator

df = pd.read_csv('./data/ground/paper_dic24/prova_normale_with_Tamb_offset.csv')
lookup = pd.read_csv('./data/config/params/lookup_cell_1exp.csv')

df = df[5000:]

#print(type(lookup[['soc', 'temp']]))
#exit()

#x_points = [[l[i] for l in lookup[['soc', 'temp']].values] for i in range(len(lookup))]
x_points = lookup[['soc', 'temp']].values
y_values = lookup['voc'].values
ocv_interp = LinearNDInterpolator(points=x_points, values=y_values)

tolerance = 1e-6
max_iter = 1000
it = 0
correction_factor = 0.01
soc_it = 0.8

while it < max_iter:
    ocv_estimated = ocv_interp(soc_it, df['Temp(degC)'].iloc[0])
    error = ocv_estimated - df['Voltage(V)'].iloc[0]
    
    if abs(error) < tolerance:
        break

    soc_it = soc_it - correction_factor * error
    it += 1


#ocv_estimated = ocv_interp(0.69, df['Temp(degC)'][0])

print(ocv_estimated, df['Voltage(V)'].iloc[0])
print('iter: {} - {}'.format(it, soc_it))