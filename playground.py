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

# ===================================================================
# ++++++++++++++++++++++ Acquisition function +++++++++++++++++++++++
# ===================================================================

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


mu_s, cov_post = gp_posterior(X_train=X_train, y_train=y_train, X_test=X_test, ell=ell, sigma=sigma, noise_std=noise_std )
std_s = posterior_std(cov_post)
ucb_vals = acquisition_ucb(mu_s, std_s, 1.76549)
next_idx = [np.argmax(ucb_vals)]

print(f"\n\t------------------------------")
print(f"MU_S:\t{mu_s}\n COV_POST:\t{cov_post}")
print(f"\nSTD:\t{std_s}")
print(f"\nUBC VALUES:\t{ucb_vals}")
print(f"\nNEST_IDX:\t{next_idx}")
print(f"\n\t------------------------------")





# mean, cov_post = gp_posterior(X_train=X_train, y_train=y_train, X_test=X_test, ell=ell, sigma=sigma, noise_std=noise_std)
# print(f"\nHere is the mu star from the gp_posterior function:\n{mean}")

# print(f"\nHere is the posterior covariance from the gp_posterior function:\n{cov_post}")

# print(f"\nHere is the diagonal using np.diag(cov_post): {np.diag(cov_post)}")

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
