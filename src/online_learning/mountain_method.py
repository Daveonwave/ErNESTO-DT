import numpy as np
from scipy.spatial.distance import cdist

class MountainCluster:
    def __init__(self, grid_size=0.1, sigma=1.0):
        self.grid_size = grid_size
        self.sigma = sigma
        self.grid_points = None
        self.potential = None
        self.peaks = None
        self.peak_assignments = None

    def _compute_mountain_function(self, data):
        x_min, x_max = min(data[:, 0]), max(data[:, 0])
        y_min, y_max = min(data[:, 1]), max(data[:, 1])
        z_min, z_max = min(data[:, 2]), max(data[:, 2])

        # Create 3D grid points
        grid_x, grid_y, grid_z = np.mgrid[x_min:x_max:self.grid_size, y_min:y_max:self.grid_size, z_min:z_max:self.grid_size]
        self.grid_points = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]

        self.potential = np.zeros(len(self.grid_points))

        for point in data:
            distances = cdist(self.grid_points, [point])
            self.potential += np.exp(-distances ** 2 / (2 * self.sigma ** 2)).ravel()

    def _find_peaks(self, threshold):
        peaks = []
        while self.potential.max() > threshold:
            peak_idx = np.argmax(self.potential)
            peak_point = self.grid_points[peak_idx]
            peaks.append(peak_point)

            # Suppress the surrounding potential
            distances = cdist(self.grid_points, [peak_point])
            self.potential -= np.exp(-distances**2 / (2 * self.sigma**2)).ravel()

        self.peaks = np.array(peaks)
        return self.peaks

    def _assign_data_to_peaks(self, data):
        distances = cdist(data, self.peaks)
        self.peak_assignments = np.argmin(distances, axis=1)

    def _find_largest_peak_data(self, data):
        largest_peak_idx = np.argmax(np.bincount(self.peak_assignments))
        largest_peak_data_indices = np.where(self.peak_assignments == largest_peak_idx)[0]
        return data[largest_peak_data_indices]

    def mountain_method(self, data, trashold):
        self._compute_mountain_function(data)
        self._find_peaks(trashold)
        self._assign_data_to_peaks(data)

        return  self._find_largest_peak_data(data)





