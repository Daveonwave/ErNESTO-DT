from scipy.spatial import distance
import numpy as np
import math


def spatio_temporal_distance(datapoint_one, datapoint_two, p):
    dist = distance.euclidean(datapoint_one, datapoint_two)
    # todo: understand how to implement the temporal term: args index_one, index_two, penalization and i
    return dist/(2*p)


def omega_rmm(datapoint, center, radius, p):
    return np.exp(-spatio_temporal_distance(datapoint_one=datapoint, datapoint_two=center,
                                            p=p)/(2*radius**2))


def largest_cluster(centers, epsilon, outliers):
    sets = []
    centers_copied = centers.copy()

    while centers_copied:
        c_j = centers_copied.pop(0)
        current_set = [centers.index(c_j)]
        centers_to_remove = []

        for idx, c_h in enumerate(centers_copied):
            if spatio_temporal_distance(c_j, c_h, p=1) <= 2 * epsilon:
                current_set.append(idx)
                centers_to_remove.append(c_h)

        for c_h in centers_to_remove:
            centers_copied.remove(c_h)

        sets.append(current_set)

    clusters = []
    for index_set in sets:
        clusters.append([outliers[idx] for idx in index_set])

    largest_cluster = max(clusters, key=len)

    for idx, elem in enumerate(largest_cluster):
        largest_cluster[idx] = np.array(elem)

    return largest_cluster


def mountain_method(outliers, epsilon, radius, p):
    err = math.inf
    centers = outliers.copy()
    while err >= epsilon:
        # assuming each outlier-point is a candidate center -> optimality property
        centers_old = centers.copy()
        for i, c_j in enumerate(centers):

            num = sum(np.array(theta_h) * omega_rmm(datapoint=theta_h, center=c_j,
                                                    p=p, radius=radius) for theta_h in outliers)
            den = sum(omega_rmm(datapoint=theta_h, center=c_j,
                                p=p, radius=radius) for theta_h in outliers)
            c_j = num / den
            centers[i] = c_j
            zipped = zip(centers, centers_old)

        err = max(distance.euclidean(c_j, c_j_old) for c_j, c_j_old in zipped)

    return centers
