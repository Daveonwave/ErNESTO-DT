import logging
import operator
import pandas as pd
from tqdm import tqdm

from . import BaseSimulator 
from src.digital_twin.orchestrator import DataWriter
from src.digital_twin.orchestrator import ScheduledLoader
from src.digital_twin.bess import BatteryEnergyStorageSystem


logger = logging.getLogger('ErNESTO-DT')


class ScheduledSimulator(BaseSimulator):
    """
    Simulator of the experiment where a schedule of instructions is provided.    
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 data_loader: ScheduledLoader,
                 data_writer: DataWriter,
                **kwargs
                 ):
        """
        Args:
            model_config (dict): configuration of the battery models
            sim_config (dict): configuration of the simulation
            data_loader (ScheduledLoader): instance of the loader tasked to provide the instructions
            data_writer (DataWriter): instance of the writer tasked to save the data
        """
        self._mode = "scheduled"
        logger.info("Instantiated the {} experiment to simulate a schedule of instructions.".format(self.__class__.__name__))

        super().__init__()
        
        # Simulation variables
        self._sample = None
        self._k = 0
        self._elapsed_time = 0
        self._done = False
        
        # Instantiate the BESS environment
        self. _battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery'],
            check_soh_every=sim_config['check_soh_every'] if 'check_soh_every' in sim_config else None
        )
        
        self._instructions = []
        # TODO: Implement the event system
        self._events = {
            'overvoltage': [],
            'undervoltage': [],
            'overheat': []
        }
        
        self._loader = data_loader
        self._writer = data_writer
        
    def _init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Scheduled Simulation' started...")
        self._battery.reset()
        self._battery.init()
        self._k = 0
        
    def _run(self):
        """
        TODO: differentiate the run method from the solve method.
        """
        self.solve()
        
    def _solve(self):
        """
        
        """
        self.init()

        pbar = tqdm(total=len(self._loader), position=0, leave=True)
        
        instructions = self._loader.collection()

        # Main loop of the simulation
        for _ in range(len(self._loader)):
            cmd = next(instructions)            
            event_start = self._elapsed_time
            logger.info("Starting command: " + cmd['sentence'])

            if 'duration' in cmd and 'until_cond' not in cmd:
                self._run_for_time(load=list(cmd['load'].keys())[0],
                                   value=list(cmd['load'].values())[0],
                                   time=cmd['duration']
                                   )

            elif 'until_cond' in cmd and 'duration' not in cmd:
                self._run_until_cond(load=list(cmd['load'].keys())[0],
                                     value=list(cmd['load'].values())[0],
                                     cond_var=list(cmd['until_cond'].keys())[0],
                                     cond_value=list(cmd['until_cond'].values())[0],
                                     action=cmd['action']
                                     )

            elif 'until_cond' in cmd and 'duration' in cmd:
                self._run_for_time_or_cond(load=list(cmd['load'].keys())[0],
                                           value=list(cmd['load'].values())[0],
                                           cond_var=list(cmd['until_cond'].keys())[0],
                                           cond_value=list(cmd['until_cond'].values())[0],
                                           time=cmd['duration'],
                                           action=cmd['action']
                                           )
            else:
                logging.error("The experiment configuration is not feasible or not implemented yet")
                exit(1)

            pbar.update(1)
            logger.info("Command executed!")
            self._instructions.append([event_start, self._elapsed_time])

        pbar.close()
        logger.info("'Scheduled Simulation' ended without errors!")
        self.done = True
            
    def _run_for_time(self, load: str, value: float, time: float):
        """
        Run the current instruction for a certain amount of time.

        Args:
            load (str): load variable to be used
            value (float): value of the load variable
            time (float): duration of the instruction
        """
        duration = self._elapsed_time + time
        self._battery.load_var = load

        while self._elapsed_time < duration:
            self._sample = {'time': self._elapsed_time, 'load': load, 'value': value}
            self.step()
            self._store_step()

    def _run_until_cond(self, load: str, value: float, cond_var: str, cond_value: float, action: str):
        """
        Run the current instruction until a condition is met.

        Args:
            load (str): load variable to be used
            value (float): value of the load variable
            cond_var (str): variable whose the condition is based on
            cond_value (float): value of the cond_var
            action (str): action occurring in the battery (charge or discharge)
        """
        self._battery.load_var = load

        curr_value = self._battery.get_i if cond_var == 'current' else self._battery.get_v
        op = operator.lt if action == 'charge' else operator.gt
        
        while op(curr_value(), cond_value):
            self._sample = {'time': self._elapsed_time, 'load': load, 'value': value}
            self.step()
            self._store_step()

    def _run_for_time_or_cond(self, load: str, value: float, cond_var: str, cond_value: float, time: float, action: str):
        """
        Run the current instruction until a condition is met or a certain amount of time has passed.

        Args:
            load (str): load variable to be used
            value (float): value of the load variable
            cond_var (str): variable whose the condition is based on
            cond_value (float): value of the cond_var
            time (float): duration of the instruction
            action (str): action occurring in the battery (charge or discharge)
        """
        duration = self._elapsed_time + time
        self._battery.load_var = load

        curr_value = self._battery.get_i if cond_var == 'current' else self._battery.get_v
        op = operator.lt if action == 'charge' else operator.gt
        
        while op(curr_value(), cond_value) and self._elapsed_time < duration:
            self._sample = {'time': self._elapsed_time, 'load': load, 'value': value}
            self.step()
            self._store_step()
            
    def _step(self):
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
        self._battery.step(load=self._sample['value'], dt=self._loader.timestep, k=self._k)
        self._battery.t_series.append(self._elapsed_time)
        self._elapsed_time += self._loader.timestep
        self._k += 1
            
    def _stop(self):
        """
        Pause the interactive simulation.
        """
        pass
    
    def _store_sample(self):
        """
        Add the ground and simulated data to the writer queues.
        """
        self._writer.add_ground_data(self._sample)
        self._writer.add_simulated_data(self._battery.get_snapshot())
    
    def _close(self):
        """
        Quit every instance of the current simulation.
        """
        self._loader.destroy()
        self._writer.stop()
        self._writer.close()
        del self._battery
    
    