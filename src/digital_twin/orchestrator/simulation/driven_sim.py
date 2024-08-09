import pandas as pd
import logging
from tqdm import tqdm

from . import BaseSimulator
from src.digital_twin.orchestrator import DrivenLoader
from src.digital_twin.orchestrator import DataWriter
from src.digital_twin.bess import BatteryEnergyStorageSystem

logger = logging.getLogger('ErNESTO-DT')


class DrivenSimulator(BaseSimulator):
    """
    Simulator of the experiment where a driven profile is provided.
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 data_loader: DrivenLoader,
                 data_writer: DataWriter
                 ):
        self._mode = "driven"
        logger.info("Instantiated the {} experiment to simulate a specific profile.".format(self.__class__.__name__))

        super().__init__()
        
        # Simulation variables
        self._sample = None
        self._get_rest_after = sim_config['get_rest_after'] if 'get_rest_after' in sim_config and sim_config['get_rest_after'] is not None else 3600
        self._elapsed_time = -1
        self._done = False
        
        # Instantiate the BESS environment
        self. _battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery'],
            check_soh_every=sim_config['check_soh_every'] if 'check_soh_every' in sim_config else None
        )
        
        self._loader = data_loader
        self._writer = data_writer
        
    def solve(self):
        """
        Execute the entire simulation from the start to the end.
        """
        logger.info("'Driven Simulation' started...")
        self._battery.reset()
        self._battery.init()
        self._battery.load_var = self._loader.input_var
        self._writer.add_simulated_data(self._battery.get_status_table())

        k = 0
        dt = self._loader.timestep if self._loader.timestep is not None else 1
        prev_time = -1
        pbar = tqdm(total=int(self._loader.duration), position=0, leave=True)
        
        inputs = self._loader.collection()
        self._sample = next(inputs)
        
        # Main loop of the simulation
        while not self._done:
            
            # Make a step in the simulation. If dt == 0 then no progresses have been made.
            if dt != 0:
                k, dt = self.step(k=k, dt=dt)    
                pbar.update(dt)
            
            #print(self._sample['time'], self._battery.t_series[-1], dt, self._elapsed_time)
            
            # Check if the simulation is over, otherwise get the next sample.
            if self._elapsed_time < self._loader.duration:
                prev_time = self._sample['time']
                self._sample = next(inputs)
                dt = round(self._sample['time'] - prev_time, 2)
            else:
                self._done = True
            
        pbar.close()
        logger.info("'Driven Simulation' ended without errors!")
            
    def step(self, k: int, dt: float):
        """
        Execute a step of the simulation that can have a fixed or variable timestep.
        
        If the timestep is variable, when the dt overcomes the step size it means that there 
        could be a lack of data in load profile, thus we consider the battery as turned off 
        for (dt-1) seconds by adding a "fake" instruction to rest the battery.
        
        Otherwise, if the timestep is fixed, the simulation progresses with the provided step size.
        
        Args:
            k (int): current iteration of the simulation
            dt (float): delta of time between the current and the previous sample
        """
        # If the timestep is variable and the dt is too large, then rest the battery.
        if self._loader.timestep is None and dt > self._get_rest_after:
            self._battery.load_var = 'current'
            self._battery.step(load=0, dt=dt-1, k=k)
            self._battery.load_var = self._loader.input_var
            
            self._elapsed_time += (dt - 1)
            self._battery.t_series.append(self._elapsed_time)
            dt = 1
            k += 1
            
            self._writer.add_simulated_data(self._battery.get_status_table())
            
        # Normal operating step of the battery system.
        ground_temp = self._sample['temperature'] if 'temperature' in self._sample else None
        self._battery.step(load=self._sample[self._loader.input_var], dt=dt, k=k, ground_temp=ground_temp)
        
        self._elapsed_time += dt
        self._battery.t_series.append(self._elapsed_time)
        k += 1
        
        self._writer.add_ground_data(self._sample)
        self._writer.add_simulated_data(self._battery.get_status_table())
        
        return k, dt    
    
    def stop(self):
        """
        Pause the interactive simulation.
        """
        pass
    
    def close(self):
        """
        Quit every instance of the current simulation.
        """
        self._loader.destroy()
        self._writer.stop()
        self._writer.close()
        del self._battery
    