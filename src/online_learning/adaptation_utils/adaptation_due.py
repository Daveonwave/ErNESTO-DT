from scipy import stats
from src.online_learning.adaptation_utils import minimum_covariance_determinant as mcd, mountain_method as mm

def fault_cluster_creation(cluster_points, outliers):

    cluster_points_flattened = cluster_points.flatten()
    outliers_flattened = outliers.flatten()

    ks_value, p_value = stats.ks_2samp(cluster_points_flattened, outliers_flattened)
    alpha_c = 0.05

    print("ks_value",ks_value,"p_value",p_value)

    if p_value < alpha_c:
        print("Reject the null hypothesis: f_hat and f_tau are different. Create a new cluster.")
        mc = mm.MountainCluster()

        # TODO: UNDERSTAND HOW TO SET THE TRASHOLD
        O_tilde = mc.mountain_method(data=outliers, threshold=0.3)
        print("len di O_tilde:",len(O_tilde))

        phi = mcd.create_cluster(O_tilde, p=10, support_fraction=0.65, variance_threshold=0.70)
        return phi
    else:
        print("Fail to reject the null hypothesis: f_hat and f_tau are the same. No new cluster needed.")
        return None