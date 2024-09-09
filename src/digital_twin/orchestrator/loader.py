from src.preprocessing.data_preparation import load_data_from_csv, sync_data_with_step
from src.preprocessing.schedule.schedule import Schedule
import logging

logger = logging.getLogger('ErNESTO-DT')


class DataLoader:
    """
    Class handling the functions to collect data from input file or stream and provide
    data to the simulator.
    """
    INPUT_VARS = ['power', 'current', 'voltage']
    
    @classmethod
    def get_instance(cls, mode: str):
        """
        Get the instance of the subclass for the current experiment mode, checking if the mode name is
        contained inside the subclass name.
        NOTE: this works because of the __init__.py, otherwise the method __subclasses__() cannot find
              subclasses in other not yet loaded modules.
        """
        if mode == 'schedule':
            return ScheduledLoader
        elif mode == 'ground_data':
            return DrivenLoader
        elif mode == 'stream':
            return StreamLoader
        else:
            raise KeyError("The chosen input data mode is not existent!")
    
    def __len__(self):
        raise NotImplementedError
        
    def collection(self):
        raise NotImplementedError
    
    def get_all_data(self):
        raise NotImplementedError
    
    def destroy(self):
        raise NotImplementedError


class ScheduledLoader(DataLoader):
    """
    Loader of an experiment based on a predefined schedule.
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self._schedule = Schedule(instructions=config['input']['schedule']['instructions'], 
                                  c_value=config['input']['schedule']['constants']['nominal_capacity'],
                                  n_cycles=config['input']['cycle_for'])
        self._timestep = config['timestep'] if 'timestep' in config and config['timestep'] is not None else 1
    
    @property
    def timestep(self):
        return self._timestep
    
    def __len__(self):
        """
        Return the current schedule buffer size.
        """
        return len(self._schedule)

    def collection(self):
        """
        Get current scheduled event and pops it from the list of events. The generation function
        yields one event at time.
        """
        while not self._schedule.is_empty():
            cmd = self._schedule.get_cmd()
            self._schedule.next_cmd()
            yield cmd
            
    def get_all_data(self):
        return self._schedule.buffer
    
    def destroy(self):
        del self._schedule


class DrivenLoader(DataLoader):
    """
    Loader of data of experiments which are driven by a specific profile.
    """
    def __init__(self, config: dict):
        super().__init__()

        self._input_var = config['input']['ground_data']['load']
        assert self._input_var in self.INPUT_VARS, "Provided loaded variables is not compliant "  \
            "with the simulator settings."
        self._ground_vars = [item['var'] for item in config['input']['ground_data']['vars']]
        
        

        self._times, self._data = (
                load_data_from_csv(csv_file=config['input']['ground_data']['file'],
                                   vars_to_retrieve=config['input']['ground_data']['vars'],
                                   time_format=config['input']['ground_data']['time_format'],
                                   iterations=config['iterations'] if 'iterations' in config else None)
                )
        
        if 'timestep' in config and config['timestep'] is not None:
            self._timestep = config['timestep']
            self._times, self._data = sync_data_with_step(times=self._times.copy(),
                                                          data=self._data.copy(),
                                                          sim_step=self._timestep)
        else:
            self._timestep = None
        
        self._data['time'] = [t - self._times[0] for t in self._times]
        self._duration = self._times[-1] - self._times[0]
        
        # If the input data has to be repeated for multiple cycles then the load_var and time are extended
        if 'cycle_for' in config['input']['ground_data'] and config['input']['ground_data']['cycle_for'] > 1:
            if len(self._ground_vars) > 1:
                logger.warning("If you want to repeat the input data for multiple consequent times, " \
                                "the variables other than the load one will become meaningless and " \
                                "will be dropped.")
            
            # Extend the input data for the number of cycles, the other variables are dropped
            self._data[self._input_var] = self._data[self._input_var] * config['input']['ground_data']['cycle_for']
            self._data['time'].extend([t + self._duration * i for i in range(1, config['input']['ground_data']['cycle_for']) for t in self._data['time']])
            
            self._data = {self._input_var: self._data[self._input_var], 'time': self._data['time']}
            self._duration = self._duration * config['input']['ground_data']['cycle_for']
        
    @property
    def input_var(self):
        return self._input_var
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def timestep(self):
        return self._timestep
    
    @property
    def ground_vars(self):
        return self._ground_vars
    
    def __getitem__(self, var:str):
        return self._data[var]    

    def __len__(self):
        """
        Return the duration in time and the number of samples within the data collection.
        """
        return len(self._data[self._input_var])
    
    def collection(self):
        """
        Yields i-th time instant and data read from csv
        """
        for i in range(len(self._data[self._input_var])):
            yield {key: self._data[key][i] for key in self._data.keys()}

    def get_all_data(self):
        return self._data
    
    def destroy(self):
        del self._data
        del self._times


class StreamLoader(DataLoader):
    """
    
    """
    def __init__(self, config: dict):
        super().__init__()
        
    # TODO: with the stream data need to be append at each new sample or at each new batch
