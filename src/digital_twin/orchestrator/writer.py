import os
import time
import threading
import queue
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger('ErNESTO-DT')


class DataWriter:
    """
    Class that writes the data to a csv file in a thread-safe way.
    
    The split in multiple files is done when the file size exceeds the maximum size, but it can happen that the
    saved file overcome the maximum size because the data is written in chunks. Thus, the threshold could be not
    strictly respected and it's just a way to avoid huge files.
    """
    def __init__(self, output_folder: str):
        """
        Args:
            output_folder (str): path to the folder where the csv files will be saved.
        """
        super().__init__()
        self._output_folder = output_folder
        self._ground_queue = queue.Queue()
        self._sim_queue = queue.Queue()
        self._aging_queue = queue.Queue()
        self._stop_event = threading.Event()
        
        self._save_output_every = 10000
        
        # Options to split huge csv files
        self._max_csv_size = 1e9
        self._ground_counter = 0
        self._sim_counter = 0
        
        try:
            os.makedirs(self._output_folder, exist_ok=True)
        except NotADirectoryError as e:
            logger.error("It's not possible to create directory {}: {}".format(self._output_folder, e.args))
            
        self._threads = []
        self.start()
        
    def start(self):
        """
        Start the threads that write data to the csv file.
        """
        self._threads.append(threading.Thread(target=self._run_writer, args=(self._ground_queue, 'ground')))
        self._threads.append(threading.Thread(target=self._run_writer, args=(self._sim_queue, 'simulated')))
        
        for thread in self._threads:
            thread.start()

    def _run_writer(self, _queue: queue.Queue, queue_type: str):
        """
        Thread that writes the data to the csv file.

        Args:
            queue (queue.Queue): queue that has to store the data to be written to the csv file.
        """
        while not self._stop_event.is_set():
            try:
                if _queue.qsize() < self._save_output_every:
                    time.sleep(0.1)
                    continue
                else:
                    data = [_queue.get() for _ in range(self._save_output_every)]
                    _queue.task_done()
                    self._write_to_csv(data, data_type=queue_type)
                    
            except queue.Empty:
                continue
        
        # Save the remaining data
        data = [_queue.get() for _ in range(_queue.qsize())]
        _queue.task_done()
        self._write_to_csv(data, data_type=queue_type)
        
    def _write_to_csv(self, data, data_type:str='ground'):
        """
        Write data to a csv file.
        The split in multiple files is done when the file size exceeds the maximum size, but it can happen that the
        saved file overcome the maximum size because the data is written in chunks. Thus, the threshold could be not
        strictly respected and it's just a way to avoid huge files.
        
        Args:
            data (dict): data to be written to the csv file
            data_type (str): type of data to be written. Defaults to 'ground'.
        """
        df = pd.DataFrame.from_records(data)
            
        if data_type == 'ground':
            filename = 'ground_{}.csv'.format(self._ground_counter)
            
            if os.path.exists(self._output_folder / filename) and os.path.getsize(self._output_folder / filename) > self._max_csv_size:
                # Split the file        
                self._ground_counter += 1
                filename = 'ground_{}.csv'.format(self._ground_counter)
            
        elif data_type == 'simulated':
            filename = 'dataset_{}.csv'.format(self._sim_counter)
            
            if os.path.exists(self._output_folder / filename) and os.path.getsize(self._output_folder / filename) > self._max_csv_size:
                # Split the file        
                self._sim_counter += 1
                filename = 'dataset_{}.csv'.format(self._sim_counter)
            
        else:
            raise ValueError("Data type not recognized.")
        
        df.to_csv(self._output_folder / filename, 
                    mode='a', 
                    header=not pd.io.common.file_exists(self._output_folder / filename), 
                    index=False)
    
    def add_ground_data(self, data):
        self._ground_queue.put(data)
        
    def add_simulated_data(self, data):
        self._sim_queue.put(data)
            
    def stop(self):
        self._stop_event.set()
        
    def close(self):
        """
        Wait for the threads to finish and then delete the queues.
        """
        for thread in self._threads:
            thread.join()
            
        del self._ground_queue
        del self._sim_queue
        