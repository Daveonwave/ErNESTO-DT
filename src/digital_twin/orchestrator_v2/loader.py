from src.preprocessing.data_preparation import load_data_from_csv, sync_data_with_step


class DataLoader:
    """
    Class handling the functions to collect data from input file or stream and provide
    data to the simulator.
    """
    SRC_TYPES = ['csv', 'stream']
    
    def __init__(self, src_type: str, config: dict):
        """
        Args:
            src_type (str): type of source from which data should be read
            config (dict): parameters for configuring the selected data source
        """
        self._src_type = src_type
        assert self._src_type in self.SRC_TYPES, "The selected type of input data is not supported."
        
        self._data = None
        self._times = None
        self._cmds = None
        
        # Read the data from csv file
        if self._src_type == 'csv':
            self._times, self._data = (
                load_data_from_csv(csv_file=config['file_path'],
                                   vars_to_retrieve=config['vars'],
                                   time_format=config['time_format'],
                                   iterations=config['iterations'])
                )
            
            if 'stepsize' in config and config['stepsize'] is not None:
                self._times, self._data = sync_data_with_step(times=self._times.copy(),
                                                              data=self._data.copy(),
                                                              sim_step=config['stepsize'])
            self._data['time'] = [t - self._times[0] for t in self._times]
            
        else:
            ... # TODO: implement stream of data
            
    def get_length(self):
        """
        Return the duration in time and the number of samples within the data collection.
        If the source type is 'stream' this returns -1 since it could be potentially infinite.
        """
        if self._src_type == 'csv':
            duration = self._times[-1] - self._times[0]
            return duration, len(self._data)
        else:
            return -1
        
    def iterator(self):
        """
        Iterate over the data collection and return the next element
        """
        if self._src_type == 'csv':
            return self._pop_from_csv()
        else:
            return self._pop_from_stream()
    
    def _pop_from_csv(self):
        """
        Yields i-th time instant and data read from csv
        """
        for time, datum in zip(self._times, self._data):
            yield time, datum
    
    def _pop_from_stream(self):
        pass
    
    # TODO: with the stream data need to be append at each new sample or at each new batch
    
    def get_data(self):
        return self._data
    