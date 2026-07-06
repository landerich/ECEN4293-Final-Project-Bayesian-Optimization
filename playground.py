import numpy as np
import scipy as sp


# ====================================================================
# ====================================================================
# ++++++++++++++++++++++++ Testing Values ++++++++++++++++++++++++++++

rng = np.random.default_rng()

test_vector1 = rng.uniform(low=-4, high=4, size=4)
test_vector2 = rng.uniform(low = -7, high = 7, size = 3)

X_train = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 0.95])
X_test = np.linspace(0.0, 1.0, 100)
y_train = np.array([0.00, 0.78, 0.81, -0.28, -1.02, -0.18])

ell = 0.25
sigma = 1.0
noise_std = 0.08

# ====================================================================
# ====================================================================

np.set_printoptions(precision=2, suppress=True)


def squared_exponential_kernel(x1: float, x2: float, length_scale: float = 1.0, sigma: float = 1.0) -> float:
    """ 
    Kernel function: Covariance function that returns the scalar covariance value.
    ----------
    Args:
        x1: Scalar input 1
        x2: Scalar input 2
        length_scale: 
        sigma: signal variance / amplitude

    Returns:
        float: Covariance between x1 and x2 under the squared sum exponential kernel.

    """
    cov = (x1 - x2) ** 2
    lgt = 2*(length_scale**2)
    
    return sigma**2 * np.exp(-(cov/lgt))

def build_covariance_matrix(arr1, arr2, ell, sigma): 
    """
    Covariance Matrix: Returns a kernel matrix n x m (len(arr1) x len(arr2)).
    ----------
    Args:
        arr1: Array of data points to be correlated to arr2 (1D input for now, to be extended to 2D eventually)
        arr2: Array of data points to be correlated to arr1 (1D input for now, to be extended to 2D eventually)

    Returns:
        float: n x m matrix expressing the covariance of arr1 and arr2

    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    n = len(arr1)
    m = len(arr2)
    cov = np.zeros((n, m))

    for i, x1_i in enumerate(arr1):
        for j, x2_j in enumerate(arr2):
            cov[i, j] = squared_exponential_kernel(x1_i, x2_j, ell, sigma)

    return cov


def gp_posterior(X_train, y_train, X_test, ell: float, sigma: float, noise_std: float):

    """
    Gaussian Process Posterior:
    ----------
    Args:
        X_tn: Shape (n,) for current 1D version.

    Returns:
        Mean of shape (m,) and covariance shape (m, m
    
    """

    X_tn = np.asarray(X_train)
    X_tt = np.asarray(X_test)
    Y_tn = np.asarray(y_train)

    # if (len(X_tn) != len(Y_tn)):    # Verify that we have a training point x for every y, change for a more robust approach whenever expanding to multiple dimensions

    #     raise ValueError("Matrices X_train and Y_train do not have the same dimensions" \
    #     "try with two matrices that have the same dimensions.")

    K_xx = build_covariance_matrix(X_tn, X_tn, ell, sigma)
    K_xs = build_covariance_matrix(X_tn, X_tt, ell, sigma)
    K_ss = build_covariance_matrix(X_tt, X_tt, ell, sigma)
    
    n = len(X_tn)

    C = K_xx + noise_std**2 * np.eye(n)

    alpha = np.linalg.solve(C, Y_tn) # For development solve is fine, but later on when improving this code swap to cholesky

    V = np.linalg.solve(C, K_xs)

    mu_s = K_xs.T @ alpha

    correction_term = K_xs.T @ V

    cov_posterior = K_ss - correction_term

    return mu_s, cov_posterior

mean, cov_post = gp_posterior(X_train=X_train, y_train=y_train, X_test=X_test, ell=ell, sigma=sigma, noise_std=noise_std)
print(f"\nHere is the mu star from the gp_posterior function:\n{mean}")

print(f"\nHere is the posterior covariance from the gp_posterior function:\n{cov_post}")

print(f"\nHere is the diagonal using np.diag(cov_post): {np.diag(cov_post)}")