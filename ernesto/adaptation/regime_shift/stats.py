import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis, cdist
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet
from scipy.stats import f, chi2


def spatio_temp_norm(points: np.ndarray, ref: np.ndarray, times: np.ndarray, ref_time: float, lambda_: float, p: int, curr_time: float):
    """
    Compute spatial-temporal normalization between an array and a given point.
    
    Args:
        points: Array of points to compare.
        ref: The reference point.
        ref_idx: The index of the reference point in the original array.
        lambda_: Weighting factor for spatial vs temporal distance.
        p: The number of dimensions (features).
        batch_size: The size of the batch (number of points).
    """
    spatial_dist = np.linalg.norm(points - ref, axis=1)**2 / (2*p)
    temporal_dist = np.abs(times - ref_time) / curr_time
    result = lambda_ * spatial_dist + (1 - lambda_) * temporal_dist
    return result


def ks_test(inliers: np.ndarray, outliers: np.ndarray):
        """
        Perform the Kolmogorov-Smirnov test between outliers and cluster samples to
        check if they belong to the same distribution. 
        
        Args:
            inliers (list): The points of the cluster.
            outliers (np.ndarray): The points of the outliers.
        
        Returns:
            D, p-value : KS test statistics and p-value.       
        """
        pca = PCA(n_components=1)       
        
        X = np.vstack([outliers, inliers])
        X_proj = pca.fit_transform(X)
        outliers_proj = X_proj[:len(outliers), 0]
        cluster_proj = X_proj[len(outliers):, 0]
        
        return ks_2samp(outliers_proj, cluster_proj)


# NON USATO
def grid_mountain_method(points: np.ndarray, 
                         times: np.ndarray,
                         grid_size: int = 20,
                         radius: float = 0.2, 
                         max_potential: float = 1e-3, 
                         lambda_: float = 0.8,
                         beta: float = 1.5, 
                         eta: float = 1e-6,
                         curr_time: int = 0
                         ):
    """
    Mountain Method to identify a cluster center from outliers.
    
    Args:
        points : np.ndarray (n_samples, n_features)
        radius   : radius of influence
        beta     : suppression parameter
        grid_size: resolution of grid per feature
    
    Returns:
        cluster_center : np.ndarray (1, n_features)
    """
    n_samples, n_features = points.shape
    
    # Normalize the batch
    norm_points = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-9)
    # norm_points = points
    #print("Normalized: ", norm_points)

    # Create grid across feature space
    mins = norm_points.min(axis=0)
    maxs = norm_points.max(axis=0)

    grid_axes = [np.linspace(mins[d], maxs[d], grid_size) for d in range(n_features)]
    mesh = np.meshgrid(*grid_axes)
    grid_points = np.vstack([m.flatten() for m in mesh]).T  # (grid_size^n_features, n_features)
    
    # Compute mountain function for each grid point
    M = np.zeros(len(grid_points))
    for i, g in enumerate(grid_points):
        norm = spatio_temp_norm(points=norm_points, ref=g, lambda_=lambda_, p=n_features, times=times, ref_time=curr_time, curr_time=n_samples)
        #norm = np.linalg.norm(points - g, axis=1)**2
        M[i] = np.sum(np.exp(-norm / (2 * radius**2)))
    
    centers = []
    while True:
    # 3. Find point with max potential
        max_idx = np.argmax(M)
        if M[max_idx] < max_potential:
            break
        
        center = grid_points[max_idx]
        centers.append(center)

        # 4. Reduce potential in neighborhood
        norm = spatio_temp_norm(points=grid_points, ref=center, lambda_=1, p=n_features, times=curr_time, ref_time=curr_time, curr_time=n_samples)
        #norm = np.linalg.norm(grid_points - center, axis=1)**2
        M -= M[max_idx] * np.exp(-norm / (2 * (beta * radius)**2))
        
    # Attribute points to centers, if they are within 2*eta
    indices = []
    for center in centers:
        norm = spatio_temp_norm(points=norm_points, ref=center, lambda_=lambda_, p=n_features, times=times, ref_time=curr_time, curr_time=n_samples)
        #print(norm)
        mask = np.sqrt(norm) < (2*eta)
        #print("Mask:", mask)
        indices.append(np.where(mask)[0])
        #print(indices)
        
    # Un-normalize centers to original scale
    centers = np.array(centers) * (points.std(axis=0) + 1e-9) + points.mean(axis=0)
    return centers, indices, grid_points, M


def trovo_mountain_method(points: np.ndarray, 
                          times: np.ndarray, 
                          curr_time: float,
                          lambda_: float = 0.8, 
                          radius: float = 0.2, 
                          eta: float = 1e-6,
                          max_iter: int = 100
                          ):
    """
    TROVO Mountain Method to identify a cluster center from outliers.
    
    Args:
        points: np.ndarray (n_samples, n_features)
        times: np.ndarray (n_samples,)
        curr_time: current time step (for temporal normalization)
        lambda_: weighting factor for spatial vs temporal distance
        radius: radius of influence
        eta: convergence threshold
        max_iter: maximum number of iterations
    
    Returns:
        center: np.ndarray (n_features,)
        index: list of indices belonging to the chosen cluster
    """
    n_samples, n_features = points.shape

    # Normalize the batch
    norm_points = (points - points.mean(axis=0)) / (points.std(axis=0) + 1e-9)

    centers = norm_points.copy()
    centers_time = times.copy()

    iter_diff = 2 * eta
    n_iter = 0
    
    # --- Iterative potential maximization ---
    while n_iter < max_iter and iter_diff > eta:
        # pairwise spatial distances squared
        norm = spatio_temp_norm(points=centers, 
                                ref=centers, 
                                lambda_=lambda_, 
                                p=n_features, 
                                times=centers_time, 
                                ref_time=centers_time[:, None], 
                                curr_time=curr_time)

        # potential matrix
        S = np.exp(-norm / (2 * radius**2))  # shape (n_samples, n_samples)

        # weighted averages for updating centers
        weights_sum = S.sum(axis=1, keepdims=True)  # shape (n_samples, 1)
        new_centers = (S @ norm_points) / weights_sum
        new_centers_time = (S @ times) / weights_sum.ravel()

        # compute max difference
        diffs = np.sqrt(spatio_temp_norm(points=new_centers, 
                                         ref=centers, 
                                         lambda_=lambda_, 
                                         p=n_features, 
                                         times=new_centers_time, 
                                         ref_time=centers_time, 
                                         curr_time=curr_time))
        iter_diff = np.max(diffs)
        #print(f"Iteration {n_iter}, max center change: {iter_diff}")

        centers, centers_time = new_centers, new_centers_time
        n_iter += 1

    # --- Agglomerate centers into clusters ---
    clusters = [[centers[0]]]
    clusters_time = [[centers_time[0]]]
    indexes = [[0]]  # indices of centers in each cluster
    n_points = [1]
    n_clusters = 1
    
    # Iterate over samples and assign to clusters
    for i in range(1, n_samples):
        incl = False
        for j in range(n_clusters):
            # Compute distance from this center to all cluster elements (other centers) and populate clusters
            dists = np.sqrt(spatio_temp_norm(points=np.array(clusters[j]), 
                                             ref=centers[i], 
                                             lambda_=lambda_, 
                                             p=n_features,      
                                             times=np.array(clusters_time[j]), 
                                             ref_time=centers_time[i], 
                                             curr_time=curr_time))
            
            # If within 2*eta of any cluster member, include the new center in that cluster
            if np.any(dists < 2 * eta):
                incl = True
                clusters[j].append(centers[i])
                clusters_time[j].append(centers_time[i])
                indexes[j].append(i)
                n_points[j] += 1
                break
            
        # New cluster of centers
        if not incl:
            clusters.append([centers[i]])
            clusters_time.append([centers_time[i]])
            indexes.append([i])
            n_points.append(1)
            n_clusters += 1

    # --- Compute final cluster centers ---
    norm_final_centers = np.array([np.mean(cluster, axis=0) for cluster in clusters])
    # Denormalize
    final_centers = [c * (points.std(axis=0) + 1e-9) + points.mean(axis=0) for c in norm_final_centers]
    
    # for i, c in enumerate(final_centers):
    #      print(f"Cluster {i}: center={c}, size={n_points[i]}, indices={indexes[i]}\n")

    # --- Choose the most numerous cluster ---
    chosen_ind = int(np.argmax(n_points))
    chosen_center = final_centers[chosen_ind]
    cluster_indices = indexes[chosen_ind]
    
    #print(f"Chosen cluster {chosen_ind} with {n_points[chosen_ind]} centers.")
    #print("Chosen indices: ", cluster_indices)

    return chosen_center, cluster_indices, final_centers, indexes
    

# NON USATO
def mcd(points: np.ndarray, support_fraction: float = 0.75):
    """
    Compute the Minimum Covariance Determinant (MCD) for robust covariance estimation.
    
    Args:
        points : np.ndarray (n_samples, n_features)
        support_fraction: fraction of points to consider as inliers
    
    Returns:
        mean : np.ndarray (n_features,)
        covariance : np.ndarray (n_features, n_features)
    """    
    mcd = MinCovDet(support_fraction=support_fraction).fit(points)
    return mcd.location_, mcd.covariance_, mcd.support_


def mcd_custom(points: np.ndarray, max_iter_factor: int = 10):
    """
    Custom implementation of the Minimum Covariance Determinant (MCD) for robust covariance estimation.
    
    Args:
        points : np.ndarray (n_samples, n_features)
        max_iter_factor: factor to determine maximum iterations based on number of samples
    """
    n, p = points.shape

    # h = max(floor((n + p + 1)/2), p + 1)
    h = max(int(np.floor((n + p + 1) / 2)), p + 1)

    # Random initial subset
    indices = np.random.choice(n, size=h, replace=False)
    indices_old = np.zeros(h, dtype=int)
    n_iter = 0

    while not np.array_equal(np.sort(indices), np.sort(indices_old)) and n_iter < max_iter_factor * n:
        # Compute mean and covariance on current subset
        subset = points[indices]
        mean = np.mean(subset, axis=0)
        cov = np.cov(subset, rowvar=False)

        # Invert covariance for Mahalanobis distance
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # If covariance is singular, add small regularization
            cov_inv = np.linalg.pinv(cov)

        # Compute Mahalanobis distance for all points
        dists = np.array([mahalanobis(x, mean, cov_inv) for x in points])

        # Keep h smallest distances
        new_indices = np.argsort(dists)[:h]

        # Update iteration state
        indices_old = indices
        indices = new_indices
        n_iter += 1
    
    return mean, cov, indices


# NON USATO
def mcd_with_center(points: np.ndarray,
                    center: np.ndarray,
                    distance: str = 'euclidean',
                    k: int = 50,
                    support_fraction: float | None = 0.75,
                    threshold_for_maha2: str = 'fisher',
                    alpha: float = 0.05,
                    max_refine_iters: int = 10
                    ):
    """
    Compute the Minimum Covariance Determinant (MCD) for robust covariance estimation.
    
    Args:
        points : pd.DataFrame (n_samples, n_features)
        center : np.ndarray (n_features,)
    """
    N, p = points.shape
    if N <= p:
        raise ValueError(f"Need more outliers than dimensions (N={N}, p={p}).")
    
    # 1) Candidate pool: k-NN around the center (fallback to all if N<k)
    
    if distance == 'maha':
        # Empirical covariance
        cov = np.cov(points, rowvar=False)
        cov_inv = np.linalg.inv(cov + np.eye(p)*1e-9)  # stabilize inverse
        
        # Mahalanobis distances to center
        d_to_center = np.array([mahalanobis(x, center, cov_inv)**2 for x in points])
    
    elif distance == 'euclidean':
        d_to_center = cdist(points, center.reshape(1, -1)).ravel()
    
    else:
        raise ValueError("Distance metric not recognized.")
    
    k_eff = min(k, N)
    cand_idx = np.argpartition(d_to_center, k_eff-1)[:k_eff]
    X_cand = points[cand_idx]
    
    # 2) Robust fit on candidate pool
    mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(X_cand)
    mu = mcd.location_
    Sigma = mcd.covariance_
    
    # tiny ridge for stability
    Sigma = Sigma + np.eye(p) * 1e-9
    
    # 3) Initial inlier set: apply threshold on candidate pool distances
    if threshold_for_maha2 == 'fisher':
        _, _, _, in_cand_mask = maha2_fisher_threshold(X=X_cand, mean=mu, cov=Sigma, n=len(X_cand), alpha=alpha)
    elif threshold_for_maha2 == 'empirical':
        _, _, _, in_cand_mask = maha2_empirical_threshold(X=X_cand, mean=mu, cov=Sigma, alpha=alpha)
    elif threshold_for_maha2 == 'chi2':
        _, _, _, in_cand_mask = maha2_chi2_threshold(X=X_cand, mean=mu, cov=Sigma, alpha=alpha)
    else:
        raise ValueError("Unknown thresholding method.")

    members_idx = cand_idx[in_cand_mask]

    # 5) Optional refinement: re-fit with absorbed members, then expand to full outliers
    for _ in range(max_refine_iters):
        X_mem = points[members_idx]
        if len(X_mem) > p:
            # re-fit MCD on current members (more stable model)
            mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(X_mem)
            mu = mcd.location_
            Sigma = mcd.covariance_ + np.eye(p) * 1e-9

            # Recompute distances for ALL outliers to allow expansion
            _, _, _, in_mem_mask = maha2_fisher_threshold(X=X_mem, mean=mu, cov=Sigma, n=len(X_mem), alpha=alpha)
            #_, _, _, in_mem_mask = maha2_chi2_threshold(X=X_mem, mean=mu, cov=Sigma, alpha=alpha)
            print("Members mask (refined): ", in_mem_mask)
            new_members_idx = np.where(in_mem_mask)[0]
            
            print("Members (refined): ", new_members_idx)

            # Stop if stable
            if set(new_members_idx.tolist()) == set(members_idx.tolist()):
                break
            members_idx = new_members_idx
    
    return mu, Sigma, members_idx


def maha2_fisher_threshold(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, n: int, alpha: float = 0.05) -> bool:
    """
    Apply the Mahalanobis distance with Fisher's exact test to determine if a point is an outlier.

    Args:
        x : np.ndarray (n_features,)
            The point to evaluate.
        mean : np.ndarray (n_features,)
            The mean of the distribution.
        cov : np.ndarray (n_features, n_features)
            The covariance matrix of the distribution.
        n : int
            The number of samples.
        alpha : float
            Significance level for the test.

    Returns:
        bool: True if the point is an outlier, False otherwise.
    """
    p = len(mean)
    if n <= p:
        raise("Not enough samples with respect to dimensions!")
    
    cov_inv = np.linalg.inv(cov + np.eye(p) * 1e-9)
    d2 = np.array([mahalanobis(x, mean, cov_inv)**2 for x in X])
    
    # F-distribution quantile
    f_val = f.ppf(1 - alpha, p, n - p) 
    
    # Threshold from the paper
    threshold = f_val * (p * (n**2 - 1)) / (n * (n - p))
    mask = d2 > threshold

    # p-value: probability of seeing a larger distance under F distribution
    f_stat = (n * (n - p) / (p * (n**2 - 1))) * d2
    p_values = 1 - f.cdf(f_stat, p, n - p)
    return d2, threshold, p_values, mask


def maha2_empirical_threshold(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Apply the Mahalanobis distance with empirical thresholding to determine if a point is an outlier.

    Args:
        x : np.ndarray (n_features,)
            The point to evaluate.
        mean : np.ndarray (n_features,)
            The mean of the distribution.
        cov : np.ndarray (n_features, n_features)
            The covariance matrix of the distribution.
        alpha : float
            Significance level for the test.

    Returns:
        bool: True if the point is an outlier, False otherwise.
    """
    p = len(mean)

    cov_inv = np.linalg.inv(cov + np.eye(p) * 1e-9)
    d2 = np.array([mahalanobis(x, mean, cov_inv) ** 2 for x in X])

    # Empirical threshold
    threshold = np.percentile(d2, 100 * (1 - alpha))

    # p-value: probability of seeing a larger distance under empirical distribution
    p_values = 1 - np.array([np.percentile(d2, 100 * (1 - alpha)) for _ in d2])
    mask = d2 > threshold

    return d2, threshold, p_values, mask


def maha2_chi2_threshold(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, alpha: float = 0.05) -> bool:
    """
    Apply the Mahalanobis distance with Chi-squared thresholding to determine if a point is an outlier.

    Args:
        x : np.ndarray (n_features,)
            The point to evaluate.
        mean : np.ndarray (n_features,)
            The mean of the distribution.
        cov : np.ndarray (n_features, n_features)
            The covariance matrix of the distribution.
        alpha : float
            Significance level for the test.

    Returns:
        bool: True if the point is an outlier, False otherwise.
    """
    p = len(mean)

    cov_inv = np.linalg.inv(cov + np.eye(p) * 1e-9)
    d2 = np.array([mahalanobis(x, mean, cov_inv) ** 2 for x in X])

    # Chi-squared threshold
    threshold = chi2.ppf(1 - alpha, df=p)
    
    mask = d2 > threshold
    p_values = 1 - chi2.cdf(d2, df=p)
    
    return d2, threshold, p_values, mask
    
    