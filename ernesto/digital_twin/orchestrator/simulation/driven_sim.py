import pandas as pd
import logging
from tqdm.rich import tqdm

from . import BaseSimulator
from ernesto.digital_twin.orchestrator import DrivenLoader
from ernesto.digital_twin.orchestrator import DataWriter
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem

logger = logging.getLogger('ErNESTO-DT')


class DrivenSimulator(BaseSimulator):
    """
    Simulator of the experiment where a driven profile is provided.
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 data_loader: DrivenLoader,
                 data_writer: DataWriter,
                 **kwargs
                 ):
        self._mode = "driven"
        logger.info("Instantiated the {} experiment to simulate a specific profile.".format(self.__class__.__name__))

        super().__init__()
        
        # Simulation variables
        self._sample = None
        self._inputs = None
        self._get_rest_after = 60
        
        # Time variables
        self._prev_time = None
        self._elapsed_time = 0
        
        # Index of the data collections
        self._k = None          
        # Number of iterations simulated
        # self._iterations = 0    

        self._done = False
        self._pbar = None
        self._clear_collections_every = sim_config['clear_collections_every'] if 'clear_collections_every' in sim_config else 50000
        
        self._loader = data_loader
        self._writer = data_writer
        
        self._loader = data_loader
        self._writer = data_writer
        
        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery'],
            check_soh_every=sim_config['check_soh_every'] if 'check_soh_every' in sim_config else None
        )

    @property
    def battery(self):
        return self._battery
    
    @property
    def sample(self):
        return self._sample
    
    @property
    def done(self):
        return self._done
    
    def _init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Driven Simulation' started...")
        self._battery.reset()
        self._battery.init(self._loader.get_initial_data(v_max=self._battery.v_max, v_min=self._battery.v_min))
        self._battery.load_var = self._loader.input_var
        
        self._k = 0
        self._prev_time = -1
        self._writer.add_simulated_data(self._battery.get_snapshot())
        
    def init_loader(self):
        """
        Initializa the generator of the input samples.
        """
        self._inputs = self._loader.collection()
    
    def load_sample(self):
        """
        Load the next sample from the loader.
        """
        self._sample = next(self._inputs)
    
    def _run(self, iterations: int, dt: float):
        """
        TODO: differentiate the run method from the solve method.
        """
        for _ in range(iterations):
            self._step(dt=dt)
            dt = self._fetch_next()
    
    def _step(self, dt: float, sample: dict, input_var: str):
        """
        Execute a step of the simulation that can have a fixed or variable timestep.
        
        If the timestep is variable, when the dt overcomes the step size it means that there 
        could be a lack of data in load profile, thus we consider the battery as turned off 
        for (dt-1) seconds by adding a "fake" instruction to rest the battery.
        
        Otherwise, if the timestep is fixed, the simulation progresses with the provided step size.
        Args:
            dt (float): delta of time between the current and the previous sample
            sample (dict): dictionary containing the current sample
            input_var (str): key of the input variable in the sample dictionary
        """
        if dt != 0:
            # Normal operating step of the battery system.
            ground_temp = sample['temperature'] if 'temperature' in sample else None
            t_amb = sample['t_amb'] if 't_amb' in sample else self.battery.temp_amb
            
            # If dt == 0 then no progresses have been made.
            if self._loader.timestep is None and dt > self._get_rest_after:
                print("Battery is resting for {} seconds.".format(dt))
                self._battery.load_var = 'current'
                self._battery.step(load=0, dt=dt-1, k=self._k, t_amb=t_amb, ground_temp=ground_temp)
                self._battery.load_var = input_var
                self._battery.t_series.append(self._elapsed_time)
                self._elapsed_time += (dt-1)
                dt = 1
                self._k += 1
                self._writer.add_simulated_data(self._battery.get_snapshot())
                        
            self._battery.step(load=sample[input_var], dt=dt, k=self._k, t_amb=t_amb, ground_temp=ground_temp)
            self._battery.t_series.append(self._elapsed_time)
            self._elapsed_time += dt
            self._k += 1
            self._store_sample()
        
    def _fetch_next(self):
        """
        Check if the simulation is over, otherwise get the next sample.
        """
        if self._elapsed_time < self._loader.duration:
            self._prev_time = self._sample['time']
            self.load_sample()
            dt = self._sample['time'] - self._prev_time
        else:
            self._done = True
            dt = 0
        return dt
    
    def fetch_next(self):
        return self._fetch_next()    
    
    def _stop(self):
        """
        Pause the interactive simulation.
        """
        pass
    
    def _solve(self):
        """
        Execute the entire simulation from the start to the end.
        NOTE: to execute the simulation from outside, keep in mind the following structure.
        """
        self.init()
        self.init_loader()
        self.load_sample()
        
        dt = self._loader.timestep if self._loader.timestep is not None else 1
        self._pbar = tqdm(total=int(self._loader.duration), position=0, leave=True)
        
        # Main loop of the simulation
        while not self._done:            
            self._step(dt=dt, sample=self._sample, input_var=self._loader.input_var)
            self._pbar.update(dt)
            dt = self._fetch_next()
            #print(self._loader.duration, self._elapsed_time, self._k)
            
            if self._k == self._clear_collections_every:
                self.clear()                
            
        self._pbar.close()
        logger.info("'Driven Simulation' solved without errors!")
    
    def _store_sample(self):
        """
        Add the ground and simulated data to the writer queues.
        """
        self._writer.add_ground_data(self._sample)
        self._writer.add_simulated_data(self._battery.get_snapshot())
    
    def clear(self):
        self._battery.clear_collections()
    
    def _close(self):
        """
        Quit every instance of the current simulation.
        """
        self._loader.destroy()
        self._writer.stop()
        self._writer.close()
        del self._battery
    