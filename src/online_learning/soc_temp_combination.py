class SocTemp:
    def __init__(self, ranges, soc, temp):
        self._ranges = ranges
        self._combinations = None
        self._current = self._equivalent_combination(soc, temp)
        self._create_combinations()


    @property
    def current(self):
        return self._current

    def _create_combinations(self):
        combinations = {}
        soc_intervals = []
        temp_intervals = []

        for key, value in self._ranges.items():
            if key.startswith('soc_interval'):
                soc_intervals.extend(value)
        print(soc_intervals)

        for key, value in self._ranges.items():
            if key.startswith('temp_interval'):
                temp_intervals.extend(value)

        index = 0
        for soc_min, soc_max in zip(soc_intervals[::2], soc_intervals[1::2]):
            for temp_min, temp_max in zip(temp_intervals[::2], temp_intervals[1::2]):
                combinations[index] = {
                    "soc_interval": (soc_min, soc_max),
                    "temp_interval": (temp_min, temp_max)
                }
                index += 1

        self._combinations = combinations

    def _equivalent_combination(self, soc, temp):
        for index, cell in self.combinations.items():
            soc_interval = cell["soc_interval"]
            temp_interval = cell["temp_interval"]
            if soc_interval[0] <= soc <= soc_interval[1] and (temp_interval[0] <= temp <= temp_interval[1]):
                return index
        return None

    def is_changed(self, soc, temp):
        new = self._eqivalent_combination(soc, temp)
        if self.current == new:
            return False
        else:
            self.current = new
            return True
