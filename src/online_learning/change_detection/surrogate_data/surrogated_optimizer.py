import numpy as np


class GenerationMechanism:
    def __init__(self, mean, cov, num_samples):
        self.mean = mean
        self.cov = cov
        self.num_samples = num_samples
        self.outliers = None

        # Initialize list to hold all results
        self.results = []
        self.current_index = 0  # Index to track which result to pop next

    def generate_data(self):
        data = np.random.multivariate_normal(self.mean, self.cov, self.num_samples)
        tmp = data.tolist()
        self.results.extend(tmp)
        return data

    def generate_outliers(self, num_outliers, scale=2):
        outlier_mean = [scale * x for x in self.mean]
        outliers = np.random.multivariate_normal(outlier_mean, self.cov, num_outliers)
        tmp = outliers.tolist()
        self.outliers = tmp
        self.results.extend(tmp)
        return outliers

    def generate_around_outliers(self, num_samples_around_outliers, scale=1):
        # Ensure outliers have been generated
        if not self.results or self.results[-1] is None:
            raise ValueError("Outliers must be generated before generating samples around them.")

        outlier_mean = np.mean(self.outliers, axis=0)
        outlier_cov = np.cov(self.outliers, rowvar=False) * scale

        samples_around_outliers = np.random.multivariate_normal(outlier_mean, outlier_cov, num_samples_around_outliers)
        tmp = samples_around_outliers.tolist()
        self.results.extend(tmp)
        return samples_around_outliers

    def pop_result(self):
        if self.current_index >= len(self.results):
            raise ValueError("No more results available to pop.")

        result = self.results[self.current_index]
        self.current_index += 1
        return result[self.current_index]