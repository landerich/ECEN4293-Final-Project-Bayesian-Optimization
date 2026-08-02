import numpy as np
import scipy as sp

# ==============================================================
#             Kernel function (squared exponential)
# ==============================================================

def squared_exponential_kernel(x1:np.ndarray,
                               x2:np.ndarray,
                               length:float = 1.0,
                               sigma_se:float = 1.0) -> float:
    """ 
    Kernel function: Covariance function that returns the scalar covariance value.
    ----------
    Args:
        x1: 2D input vector (2,)
        x2: 2D input vector (2,)
        length: Length scale
        sigma_se: signal variance / amplitude

    Returns:
        float: Covariance between x1 and x2 under the squared sum exponential kernel.

    """
    diff = x1 - x2
    distance = np.dot(diff, diff)
    lgt = 2*(length**2)
    
    return sigma_se**2 * np.exp(-(distance/lgt))

# ==============================================================
#                  Kernel function (Linear)
# ==============================================================

def linear_kernel(x1:np.ndarray,
                  x2:np.ndarray,
                  sigma_linear:float = 1.0) -> float:
    """
    
    """
    return sigma_linear**2 * np.dot(x1, x2)

# ==============================================================
#                        Covariance matrix
# ==============================================================

def build_covariance_matrix(arr1,
                            arr2,
                            kernel_function,
                            **kernel_parameters): 
    """
    Covariance Matrix: Returns a kernel matrix n x m (len(arr1) x len(arr2)).
    ----------
    Args:
        arr1: Array of data points, shape (n, d). Each row is one input vector.
        arr2: Array of data points, shape (m, d). Each row is one input vector.
        
    Returns:
        cov : n x m matrix expressing the covariance of arr1 and arr2

    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    n = len(arr1)
    m = len(arr2)
    cov = np.zeros((n, m))

    for i, x1_i in enumerate(arr1):
        for j, x2_j in enumerate(arr2):
            cov[i, j] = kernel_function(x1_i, x2_j, **kernel_parameters)

    return cov

# ==============================================================
#                           GP Posterior
# ==============================================================

def gp_posterior(X_train,
                 y_train,
                 X_test,
                 noise_std: float,
                 kernel_function,
                 **kernel_parameters):

    """
    Gaussian Process Posterior:
    ----------
    Args:
        X_train: Shape (n, d) for current 2D version.

    Returns:
        mu_s: Mean of shape (m,) and 
        cov_posterior: Covariance shape (m, m)
    
    """

    X_tn = np.asarray(X_train)
    X_tt = np.asarray(X_test)
    Y_tn = np.asarray(y_train)

    if (X_tn.shape[0] != Y_tn.shape[0]):    # Verify that we have a training point x for every y, change for a more robust approach whenever expanding to multiple dimensions

        raise ValueError("Samples X_train and y_train must have the same number of samples\n" \
        f"got {X_tn.shape[0]} and {Y_tn.shape[0]}.")

    K_xx = build_covariance_matrix(X_tn, X_tn, kernel_function, **kernel_parameters)
    K_xs = build_covariance_matrix(X_tn, X_tt, kernel_function, **kernel_parameters)
    K_ss = build_covariance_matrix(X_tt, X_tt, kernel_function, **kernel_parameters)
    
    n = len(X_tn)

    C = K_xx + noise_std**2 * np.eye(n)

    alpha = np.linalg.solve(C, Y_tn) # For development solve is fine, but later on when improving this code swap to cholesky

    V = np.linalg.solve(C, K_xs)

    mu_s = K_xs.T @ alpha

    correction_term = K_xs.T @ V

    cov_posterior = K_ss - correction_term

    return mu_s, cov_posterior

# ==============================================================
#                 Posterior Standard Deviation Vector
# ==============================================================

def posterior_std(cov_post):
    """ Returns the posterior standard deviation from a covariance Matrix. 
    Args:
    -------------------
    cov_post: Posterior covariance matrix.

    Returns:
    -------------------
    The posterior standard deviation vector.
    """
    cov_posterior = np.asanyarray(cov_post)
    diag = np.diag(cov_posterior)
    diag = np.maximum(diag, 0.0)
    return np.sqrt(diag)  # Floating point error negatives coming from the sqrt, check this later

# ==============================================================
#                        Acquisition function
# ==============================================================

def acquisition_ucb(mu, std, kappa):     # Expected improvement is the best choice, but start with UCB
    """ Returns the scoring vector of points to sample next.
    Args:
    ----------
    mu: Posterior mean vector
    std: Posterior standard deviation vector
    kappa: Scaling factor for the Standard Deviation
    
    Returns:
    ----------
    A vector of values for future sampling
    """
    kappa = float(kappa)

    mu_acquisition = np.asarray(mu)
    std_acquisition = np.asarray(std)

    if (mu_acquisition.shape != std_acquisition.shape):
        raise ValueError("Mu and STD vectors must be the same size in order to compute UCB.")

    return mu_acquisition + kappa * std_acquisition

# ==============================================================
#                        Kernel Sum
# ==============================================================

def combined_kernel_sum(xin1,
                        xin2,
                        length_se,
                        sigma_se,
                        sigma_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length= length_se, sigma_se= sigma_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sigma_linear)
    return k_se + k_linear

# ==============================================================
#                        Kernel Product
# ==============================================================

def combined_kernel_product(xin1,
                            xin2,
                            length_se,
                            sigma_se,
                            sigma_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length= length_se, sigma_se= sigma_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sigma_linear)
    return k_se * k_linear
