import numpy as np
import scipy as sp
# import array as ar

rng = np.random.default_rng()





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
# ++++++++++ Kernel function (squared exponential) +++++++++++++
# ==============================================================

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


# ===================================================================
# +++++++++++++++++++++++ GP Posterior ++++++++++++++++++++++++++++++
# ===================================================================

# Things to create/do:

# Training covariance matrix
# Add noise on its diagonal
# Build cross-covariance and test covariance
# Solve the linear system
# Compute posterior mean and posterior covariance

# Previous steps completed

# Covariance matrix

def build_covariance_matrix(arr1, arr2, ell: float, sigma: float): 
    """
    Covariance Matrix: Returns a kernel matrix n x m (len(arr1) x len(arr2)).
    ----------
    Args:
        arr1: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
        arr2: Array of data points to compute pairwise covariance matrix between two 1D input arrays.
        ell: Argument needed for the squared exppnential kernel.
        sigma: Argument needed for the squared exponential kernel.

    Returns:
        narray : n x m matrix expressing the covariance of arr1 and arr2

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

# Log marginal likelihood

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

def acq_function():

    return None

np.linalg.slogdet()



# Prioritize shifting kernel from squared exponential to constant or linear kernel


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
