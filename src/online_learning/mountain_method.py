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
        print("__________________________________________________________________________")
        print("Debug here:")
        print("Data: \n", data)
        print("Observe:")
        print("data[:,1]:", data[:, 1])

        print("Extreme of the values: ")
        x_min, x_max = data[:, 0].min(), data[:, 0].max()
        print(f"x_min: {x_min}, x_max: {x_max}")
        y_min, y_max = data[:, 1].min(), data[:, 1].max()
        print(f"y_min: {y_min}, y_max: {y_max}")
        z_min, z_max = data[:, 2].min(), data[:, 2].max()
        print(f"z_min: {z_min}, z_max: {z_max}")

        # Check if min and max values for any dimension are equal and adjust if necessary
        epsilon = 1e-6  # Small value to adjust the range
        if x_min == x_max:
            x_min -= epsilon
            x_max += epsilon
            print(f"Adjusted x_min: {x_min}, x_max: {x_max}")
        if y_min == y_max:
            y_min -= epsilon
            y_max += epsilon
            print(f"Adjusted y_min: {y_min}, y_max: {y_max}")
        if z_min == z_max:
            z_min -= epsilon
            z_max += epsilon
            print(f"Adjusted z_min: {z_min}, z_max: {z_max}")

        # Define the grid size and ensure it is an integer and greater than zero
        grid_size = int(getattr(self, 'grid_size', 10))
        if grid_size <= 0:
            grid_size = 10
        print(f"Using grid size: {grid_size}")

        # Create 1D arrays of grid points
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        z = np.linspace(z_min, z_max, grid_size)

        # Create 3D grid points using np.meshgrid and np.vstack
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        self.grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        print(f"Grid shapes: grid_x: {grid_x.shape}, grid_y: {grid_y.shape}, grid_z: {grid_z.shape}")

        if len(self.grid_points) == 0:
            print("Grid points array is empty!")
        else:
            print("Grid points array created successfully.")

        print("self.grid_points:")
        print(self.grid_points)

        self.potential = np.zeros(len(self.grid_points))

        # Calculate the potential for each grid point
        for point in data:
            distances = cdist(self.grid_points, [point])
            self.potential += np.exp(-distances ** 2 / (2 * self.sigma ** 2)).ravel()

    def _find_peaks(self, threshold):
        peaks = []
        print("potential.size", self.potential.size)
        print("what's inside potential:", self.potential)
        print("potential.max :", self.potential.max)
        print("_______________________________________________________________________________________________")
        while self.potential.size > 0 and self.potential.max() > threshold:
            peak_idx = np.argmax(self.potential)
            peak_point = self.grid_points[peak_idx]
            peaks.append(peak_point)

            # Suppress the surrounding potential
            distances = cdist(self.grid_points, [peak_point])
            self.potential -= np.exp(-distances**2 / (2 * self.sigma**2)).ravel()

        self.peaks = np.array(peaks)
        return self.peaks

    def _assign_data_to_peaks(self, data):
        if self.peaks.size == 0:
            raise ValueError("No peaks found. Adjust the threshold or check the data.")
        distances = cdist(data, self.peaks)
        self.peak_assignments = np.argmin(distances, axis=1)

    def _find_largest_peak_data(self, data):
        if self.peak_assignments is None or len(self.peak_assignments) == 0:
            raise ValueError("No peak assignments found. Check if peaks were properly assigned.")
        largest_peak_idx = np.argmax(np.bincount(self.peak_assignments))
        largest_peak_data_indices = np.where(self.peak_assignments == largest_peak_idx)[0]
        return data[largest_peak_data_indices]

    def mountain_method(self, data, threshold):
        self._compute_mountain_function(data)
        self._find_peaks(threshold)
        print(self.peaks)
        if self.peaks.size == 0:
            raise ValueError("No peaks were found with the given threshold.")
        self._assign_data_to_peaks(data)

        print("the peaks are :", self.peaks)
        print("peak assignment :", self.peak_assignments)
        return self._find_largest_peak_data(data)
