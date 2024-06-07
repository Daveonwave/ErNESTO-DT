class Grid:
    def __init__(self, grid_parameters, soc, temp):
        self.grid_parameters = grid_parameters
        self.cells = self.__generate_cells()
        self._current_cell = None
        self.__check_cell(soc, temp)

    @property
    def current_cell(self):
        return self._current_cell

    def get_max_temp(self):
        max_temp = float('-inf')
        for key, value in self.grid_parameters.items():
            if key.startswith('temp_interval'):
                temp_values = value[1::2]
                if temp_values:
                    max_temp = max(max_temp, *temp_values)
        return max_temp

    def __generate_cells(self):
        cells = {}
        soc_intervals = []
        temp_intervals = []

        # Collect all SOC intervals
        for key, value in self.grid_parameters.items():
            if key.startswith('soc_interval'):
                soc_intervals.extend(value)

        # Collect all temperature intervals
        for key, value in self.grid_parameters.items():
            if key.startswith('temp_interval'):
                temp_intervals.extend(value)

        cell_index = 0
        for soc_min, soc_max in zip(soc_intervals[::2], soc_intervals[1::2]):
            for temp_min, temp_max in zip(temp_intervals[::2], temp_intervals[1::2]):
                cells[cell_index] = {
                    "soc_interval": (soc_min, soc_max),
                    "temp_interval": (temp_min, temp_max)
                }
                cell_index += 1

        return cells

    def __check_cell(self, soc, temp):
        for index, cell in self.cells.items():
            soc_interval = cell["soc_interval"]
            temp_interval = cell["temp_interval"]
            if soc_interval[0] <= soc <= soc_interval[1] and (temp_interval[0] <= temp <= temp_interval[1]):
                #just to underline that here you set also the attribute
                self._current_cell = index
                return index
        return None

    def is_changed_cell(self, soc, temp):
    # TODO: E' UNA PEZZA
    #if temp >= self.get_max_temp():
    #      return False
      if self.current_cell == self.__check_cell(soc, temp):
        return False
      else:
        return True 