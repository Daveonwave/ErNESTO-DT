import numpy as np
from tqdm import tqdm

import shutil
import datetime
import fmpy as fmi
from pythonfmu.builder import FmuBuilder


if __name__ == '__main__':
    fmu_to_build = './fmu_mockup.py'
    dest_folder = '.'
    project_files = ['../digital_twin/bess.py']
    FmuBuilder.build_FMU(script_file=fmu_to_build, dest=dest_folder, project_files=project_files)




    filename = 'fmu/DTMockup_new.fmu'
    """
    #initial_inputs = {'soc': 0., 'temp': 0., 'delta_dod': 0., 't': 0, 'curr_iter': 0, 'change_dir': False}
    input_current = np.random.uniform(low=0, high=4, size=3600)
    init_current = 1.0

    dtype = [('time', np.double), ('load_current', np.double)]
    signals = np.array([(0., 4.), (1., 5.0), (2., 1.1), (3., 2.0)], dtype=dtype)

    output = fmi.simulate_fmu(filename=filename,
                          validate=True,
                          start_time=0,
                          step_size=1.,
                          stop_time=3600,
                          output_interval=1.0,
                          fmi_type="CoSimulation",
                          start_values={'load_current': init_current},
                          input=signals,
                          visible=True
                          )

    print(output[:10])
    """

    fmu_filename = 'fmu/DTMockup_new.fmu'
    model_description = fmi.read_model_description(fmu_filename)
    stop_time = 3600
    step_size = 1.0

    unzipdir = fmi.extract(fmu_filename)

    fmu = fmi.fmi2.FMU2Slave(guid=model_description.guid,
                             unzipDirectory=unzipdir,
                             modelIdentifier=model_description.coSimulation.modelIdentifier,
                             instanceName='instance1')

    # initialize
    fmu.instantiate()
    fmu.setupExperiment(tolerance=1E-4, startTime=0.0, stopTime=stop_time)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    start = datetime.datetime.now()

    t = 0.0
    voltage = []
    i = 0
    # simulation loop
    while t < stop_time - step_size:
        i += 1
        # perform one step
        fmu.doStep(currentCommunicationPoint=t, communicationStepSize=step_size)

        voltage.append(fmu.getReal([1])[0])
        # print("h={}".format(h))

        # advance the time
        t += step_size

    print(voltage[:10])

    end = datetime.datetime.now()
    print("sum={}, iter={}".format(sum, i))

    delta = end - start
    print("{}ms".format(int(delta.total_seconds() * 1000)))

    try:
        fmu.terminate()
    except OSError:
        pass
    fmu.freeInstance()

    # clean up
    shutil.rmtree(unzipdir)
    