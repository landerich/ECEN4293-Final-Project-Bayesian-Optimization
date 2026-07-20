import numpy as np
import scipy as sp


# ====================================================================
# ====================================================================
# ++++++++++++++++++++++++ Testing Values ++++++++++++++++++++++++++++

rng = np.random.default_rng()
test_vector1 = rng.uniform(low=-4, high=4, size=4)
test_vector2 = rng.uniform(low = -7, high = 7, size = 3)

# ==================== Set 1 of testing values ======================

X_train_one = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 0.95])
X_test_one = np.linspace(0.0, 1.0, 100)
y_train_one = np.array([0.00, 0.78, 0.81, -0.28, -1.02, -0.18])

ell_one = 0.25
sigma_one = 1.0
noise_std_one = 0.08
kappa_one = 0.532

# =================== Set 2 of testing values =======================

X_train_two = np.array([0.00, 0.12, 0.27, 0.41, 0.63, 0.84])
y_train_two = np.array([0.02, 0.66, 0.93, 0.61, -0.42, -0.95])
X_test_two = np.linspace(0.0, 1.0, 100)

ell_two = 0.20
sigma_two = 1.0
noise_std_two = 0.06
kappa_two = 2.5

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

def linear_kernel(x1: float, x2: float, sigma_linear:  float = 1.0) -> float:
    return sigma_linear**2 * x1 * x2

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

def gp_posterior(X_train, y_train, X_test, noise_std: float, kernel_function, **kernel_parameters):

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

def combined_kernel_sum(xin1, xin2, ell_se, sig_se, sig_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sig_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sig_linear)
    return k_se + k_linear

def combined_kernel_product(xin1, xin2, ell_se, sig_se, sig_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sig_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sig_linear)
    return k_se * k_linear

mu_s, cov_post = gp_posterior(X_train=X_train_two, y_train=y_train_two, X_test=X_test_two, ell=ell_two, sigma=sigma_two, noise_std=noise_std_two)
std_s = posterior_std(cov_post)
ucb_vals = acquisition_ucb(mu_s, std_s, kappa_two)
next_idx = [np.argmax(ucb_vals)]

print(f"\n\t------------------------------")
print(f"MU_S:\t{mu_s}\n COV_POST:\t{cov_post}")
print(f"\nSTD:\t{std_s}")
print(f"\nUBC VALUES:\t{ucb_vals}")
print(f"\nNEST_IDX:\t{next_idx}")
print(f"\n\t------------------------------")

# =================================================================
#  Test over 1/x function for kernel switch on asymptotic behavior
# =================================================================

def one_over_x(x):
    return 1/x

x_values = np.linspace(0, 8, 800)

y_values = one_over_x(x_values)

# =================================================================
#                        Testing function
# =================================================================

def test_bo():


    return None





# ========================= Code sample ========================


# from __future__ import division
# import numpy as np
# import matplotlib.pyplot as pl

# """ This is code for simple GP regression. It assumes a zero mean GP Prior """


# # This is the true unknown function we are trying to approximate
# f = lambda x: np.sin(0.9*x).flatten()
# #f = lambda x: (0.25*(x**2)).flatten()


# # Define the kernel
# def kernel(a, b):
#     """ GP squared exponential kernel """
#     kernelParameter = 0.1
#     sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
#     return np.exp(-.5 * (1/kernelParameter) * sqdist)

# N = 10         # number of training points.
# n = 50         # number of test points.
# s = 0.00005    # noise variance.

# # Sample some input points and noisy versions of the function evaluated at
# # these points. 
# X = np.random.uniform(-5, 5, size=(N,1))
# y = f(X) + s*np.random.randn(N)

# K = kernel(X, X)
# L = np.linalg.cholesky(K + s*np.eye(N))

# # points we're going to make predictions at.
# Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# # compute the mean at our test points.
# Lk = np.linalg.solve(L, kernel(X, Xtest))
# mu = np.dot(Lk.T, np.linalg.solve(L, y))

# # compute the variance at our test points.
# K_ = kernel(Xtest, Xtest)
# s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
# s = np.sqrt(s2)


# # PLOTS:
# pl.figure(1)
# pl.clf()
# pl.plot(X, y, 'r+', ms=20)
# pl.plot(Xtest, f(Xtest), 'b-')
# pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
# pl.plot(Xtest, mu, 'r--', lw=2)
# pl.savefig('predictive.png', bbox_inches='tight')
# pl.title('Mean predictions plus 3 st.deviations')
# pl.axis([-5, 5, -3, 3])

# # draw samples from the prior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
# f_prior = np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(2)
# pl.clf()
# pl.plot(Xtest, f_prior)
# pl.title('Ten samples from the GP prior')
# pl.axis([-5, 5, -3, 3])
# pl.savefig('prior.png', bbox_inches='tight')

# # draw samples from the posterior at our test points.
# L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
# f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
# pl.figure(3)
# pl.clf()
# pl.plot(Xtest, f_post)
# pl.title('Ten samples from the GP posterior')
# pl.axis([-5, 5, -3, 3])
# pl.savefig('post.png', bbox_inches='tight')

# pl.show()
