import numpy as np
import scipy as sp

# 1. Kernel function
# 
# 2. GP posterior (prediction)
# 
# 3. Log marginal likelyhood
# 
# 4. Acqusition function
#  
# 5. BO loop

# ==============================================================
#             Kernel function (squared exponential)
# ==============================================================

def squared_exponential_kernel(x1: float, x2: float, length_scale: float = 1.0, sigma_se: float = 1.0) -> float:
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
    
    return sigma_se**2 * np.exp(-(cov/lgt))

# ==============================================================
#                  Kernel function (Linear)
# ==============================================================

def linear_kernel(x1: float, x2: float, sigma_linear:  float = 1.0) -> float:
    return sigma_linear**2 * x1 * x2

# ==============================================================
#                        Covariance matrix
# ==============================================================

def build_covariance_matrix(arr1, arr2, kernel_function, **kernel_parameters): 
    """
    Covariance Matrix: Returns a kernel matrix n x m (len(arr1) x len(arr2)).
    ----------
    Args:
        arr1: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
        arr2: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
        ell: Argument needed for the squared exppnential kernel.
        sigma: Argument needed for the squared exponential kernel.

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

# def build_covariance_matrix(arr1, arr2, ell: float, sigma: float): 
#     """
#     Covariance Matrix: Returns a kernel matrix n x m (len(arr1) x len(arr2)).
#     ----------
#     Args:
#         arr1: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
#         arr2: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
#         ell: Argument needed for the squared exppnential kernel.
#         sigma: Argument needed for the squared exponential kernel.

#     Returns:
#         cov : n x m matrix expressing the covariance of arr1 and arr2

#     """
#     arr1 = np.asarray(arr1)
#     arr2 = np.asarray(arr2)

#     n = len(arr1)
#     m = len(arr2)
#     cov = np.zeros((n, m))

#     for i, x1_i in enumerate(arr1):
#         for j, x2_j in enumerate(arr2):
#             cov[i, j] = squared_exponential_kernel(x1_i, x2_j, ell, sigma)

#     return cov

# ==============================================================
#                           GP Posterior
# ==============================================================

def gp_posterior(X_train, y_train, X_test, ell: float, sigma: float, noise_std: float):

    """
    Gaussian Process Posterior:
    ----------
    Args:
        X_tn: Shape (n,) for current 1D version.

    Returns:
        mu_s: Mean of shape (m,) and 
        cov_posterior: Covariance shape (m, m)
    
    """

    X_tn = np.asarray(X_train)
    X_tt = np.asarray(X_test)
    Y_tn = np.asarray(y_train)

    if (len(X_tn) != len(Y_tn)):    # Verify that we have a training point x for every y, change for a more robust approach whenever expanding to multiple dimensions

        raise ValueError("Matrices X_train and Y_train do not have the same dimensions" \
        "try with two matrices that have the same dimensions.")

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

# ==============================================================
#                       Log marginal likelihood
# ==============================================================

def log_marginal(X_train, y_train, ell: float, sigma: float, noise_std: float):     # logp(y | X, theta) = -1/2 * y.T * C^-1 * y - 1/2 * (log(abs(C))) - n/2 * log(2 * pi)

    X_tn = np.asarray(X_train)
    Y_tn = np.asarray(y_train)

    K = build_covariance_matrix(X_tn, X_tn, ell, sigma)
    C = K + noise_std**2 * np.eye(len(X_tn))
    sign, logdet_c = np.linalg.slogdet(C)
    
    # If sign < = 0 flag it since is a warning that something went wrong 

    alpha = np.linalg.solve(C, Y_tn)
    quad = Y_tn.T @ alpha

    return -1/2 * quad - 1/2 * logdet_c - len(X_tn)/2 * np.log(2*np.pi)

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
    cov_posterior = np.asarray(cov_post)
    return np.sqrt(np.diag(cov_posterior))

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

def combined_kernel_sum(xin1, xin2, ell_se, sig_se, sig_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sig_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sig_linear)
    return k_se + k_linear

# ==============================================================
#                        Kernel Product
# ==============================================================

def combined_kernel_product(xin1, xin2, ell_se, sig_se, sig_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sig_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sig_linear)
    return k_se * k_linear



# Note so self: Kernel combination is a much better idea and application
# than just using a penalize and retreat approach if the BO over estimates the
# next sample point.


# If the function really behaves like 1/x, the most natural move is often to transform the inputs or 
# outputs rather than forcing a basic kernel to learn the singularity directly.


# Can the input be reparameterized so the asymptote is less singular?
# Can the output be transformed so the GP sees a smoother function?
# Do I need a composite kernel rather than a single kernel?




# Notes from Marcus:
# 1. Have the code running for next week stable and have the data visualization

# 2. Catch an error or NaN for the Kernel switch

# 3. Prioritize shifting kernel from squared exponential to constant or linear kernel
# Add to note No.3, there is no swiching, combination is better because it takes into account
# the shape and trend of the function that is being explored, yielding better results.


