import numpy as np

class Cluster:
    def __init__(self):
        self.centroid = None
        self.variance = None
        self.cluster = []

    def add(self, point):
        self.cluster.append(point)

    def get_centroid(self):
        return self.centroid

    def get_variance(self):
        return self.variance

    def compute_centroid(self):
        if self.cluster:
            self.centroid = np.mean(np.array(self.cluster),axis = 0)

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)
    # The spread of data points around a central point:
    def compute_variance(self):
        if self.cluster:
            distances = [self.euclidean_distance(self.centroid, point) for point in self.cluster]
            self.variance = np.var(distances)

