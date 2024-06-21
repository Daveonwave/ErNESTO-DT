import numpy as np
def generate_multivariate_3D_gaussian_samples(n, bounds):

    mean = [np.random.uniform(bounds['r0'][0], bounds['r0'][1]),
            np.random.uniform(bounds['rc'][0], bounds['rc'][1]),
            np.random.uniform(bounds['c'][0], bounds['c'][1])]


    A = np.random.rand(3, 3)
    cov = np.dot(A, A.transpose())  # Make it positive semi-definite

    samples = np.random.multivariate_normal(mean, cov, n) # np.ndarray: An array of shape (n, 3)
    return samples

def generate_outliers(samples, n_outliers, threshold=3):
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    outliers = []
    count = 0
    while count < n_outliers:
        point = np.random.multivariate_normal(mean, cov)
        mahalanobis_distance = np.sqrt((point - mean).T @ inv_cov @ (point - mean))
        if mahalanobis_distance > threshold:
            outliers.append(point)
            count += 1

    return np.array(outliers)


def generate_exponential_samples(n, bounds):
    samples = np.empty((n, 3))

    for i, key in enumerate(bounds):
        lower, upper = bounds[key]
        samples[:, i] = np.random.uniform(lower, upper, size=n) * np.random.exponential(1, size=n)

    return samples


def generate_norm_exp_outliers_mixed_samples(n, n_outliers, bounds_gaussian, bounds_exponential):
    # Generate the samples
    gaussian_samples = generate_multivariate_3D_gaussian_samples(n, bounds_gaussian)
    outlier_samples = generate_outliers(gaussian_samples, n_outliers)
    exponential_samples = generate_exponential_samples(n, bounds_exponential)

    # Combine the samples
    combined_samples = np.vstack((gaussian_samples, outlier_samples, exponential_samples))

    # Shuffle the combined samples
    np.random.shuffle(combined_samples)

    return combined_samples
