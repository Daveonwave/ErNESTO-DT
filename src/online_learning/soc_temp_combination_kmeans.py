class SocTemp:
    def __init__(self, clusters_ranges):
        """
        Initialize the ClusterChecker with a dictionary of clusters.

        :param clusters_ranges: A dictionary where each key is a cluster number (e.g., 0, 1, 2) and the value contains
                         'temp_interval' and 'soc_interval' for that cluster.
        """
        self.clusters_ranges = clusters_ranges
        self._current = None


    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, cluster_id):
        self._current = cluster_id

    def check_cluster(self, temp, soc):
        """
        This method takes temperature (temp) and state of charge (soc),
        and returns the cluster that matches the given values.
        """
        for cluster_id, intervals in self.clusters_ranges.items():
            temp_min, temp_max = intervals['temp_interval']
            soc_min, soc_max = intervals['soc_interval']

            if temp_min <= temp <= temp_max and soc_min <= soc <= soc_max:
                self.current = cluster_id
