###################################################################################################
# Ricardo Landeros Aranda | Oklahoma State University | Spring 2026
# ECEN 4293 Numerical Methods in Python for Engineers | Instructor: Marcus Mellor
# Final project: Bayesian Optimization in 1D for further application in D Flip-Flop c2q timing
# SPICE simulation optimization.
# This code was completed using an AI tool for learning, application, and debugging purposes.
# * BO: Bayesian Optimization
###################################################################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def generate_x0(low = 0.0, high = 100.0):
    return random.uniform(low, high)

def toy_delay_asymptote(x, x0, d_nom=50.0, scale=2000.0):
    """
    Delay model: d(x) = scale / (x - x0) for x > x0
    For x <= x0, treat as failure (NaN).
    """
    if x <= x0:
        return float('NaN')
    return d_nom + scale / (x - x0)


x0_hidden = generate_x0(low=0.0, high=50.0)

D_TARGET = 80.0 # 80 ps as an example

def delay1d(x):
    """ Vectorized wrapper over toy_delay_asymptote for BO. """
    x = np.atleast_1d(x).ravel()
    y = []
    for xi in x:
        d = toy_delay_asymptote(float(xi), x0_hidden)
        if np.isnan(d): # This is a heavy penalty if we fal into metastable/fail region
            y.append(1e6) 
        else:
            y.append(d)
    return np.array(y)

def objective_g(x):
    """
    Objective for BO: measure of how far is the delay from the target boundary D_TARGET.
    BO minimizes this.
    """
    d = delay1d(x)
    return np.abs(d - D_TARGET).reshape(-1, 1)

# Bayesian Optimization general approach code


# Objective function (example: unknown function)
def f(x):
    return np.sin(3*x) + x**2 - 0.7*x

# Expected Improvement acquisition function
def expected_improvement(X, X_sample, Y_sample, gp, xi=0.01):
    mu, sigma = gp.predict(X, return_std=True)
    
    sigma = sigma.reshape(-1, 1)
    
    y_min = np.min(Y_sample)


    with np.errstate(divide='warn'):
        Z = (y_min - mu - xi) / sigma
        ei = (y_min - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# Propose next sampling point
def propose_location(acquisition, X_sample, Y_sample, gp, bounds, n_restarts=25):
    dim = bounds.shape[0]
    best_x = None
    best_acq = -np.inf

    for _ in range(n_restarts):
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        x = x0.reshape(1, -1)
        x0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dim)
        x = x0.reshape(1, -1)
        acq_value = acquisition(x, X_sample, Y_sample, gp)

        if acq_value > best_acq:
            best_acq = acq_value
            best_x = x

    return best_x

# Bayesian Optimization loop
def bayesian_optimization(n_iters, sample_loss, bounds):

    dim = bounds.shape[0]

    X_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], (3, dim))
    Y_sample = sample_loss(X_sample[:, 0]).reshape(-1, 1)

    kernel = Matern(nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, alpha = 1e-6, normalize_y=True)

    for i in range(n_iters):
        gp.fit(X_sample, Y_sample)

        X_next = propose_location(expected_improvement, X_sample, Y_sample, gp, bounds)
        Y_next = sample_loss(X_next[:, 0]).reshape(-1, 1)

        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))

        print(f"Iteration {i+1}: x = {X_next.flatten()[0]:.4f}, g(x) = {Y_next.flatten()[0]:.4f}")

    return X_sample, Y_sample

# Run
bounds = np.array([[x0_hidden + 1.0, x0_hidden + 40.0]])  # search safely to the right of asymptote
X, Y = bayesian_optimization(15, objective_g, bounds)

best_idx = np.argmin(Y)
x_best = X[best_idx, 0]
print("Hidden x0:", x0_hidden)
print("BO-estimated boundary point x* ≈", x_best)
print("Delay at x*:", delay1d([x_best])[0])

# After:
# X, Y = bayesian_optimization(15, objective_g, bounds)
# best_idx = np.argmin(Y)
# x_best = X[best_idx, 0]

# 1) Build a dense grid and compute the true delay
x_grid = np.linspace(bounds[0, 0], bounds[0, 1], 500)
delay_grid = delay1d(x_grid)

# 2) Compute delay at BO evaluation points
delay_samples = delay1d(X[:, 0])

plt.figure(figsize=(8, 5))
# True delay curve
plt.plot(x_grid, delay_grid, label="Toy delay d(x)", color="C0")
# BO evaluation points
plt.scatter(X[:, 0], delay_samples, color="C1", label="BO samples")
# Highlight best boundary estimate
plt.scatter([x_best], [delay1d([x_best])[0]], color="red", s=80,
            label="Estimated boundary (d≈D_TARGET)")

# Draw the target delay level
plt.axhline(D_TARGET, color="gray", linestyle="--", label="D_TARGET")

plt.xlabel("x (setup-like variable)")
plt.ylabel("delay d(x)")
plt.title("Bayesian optimization on metastability-like toy delay")
plt.legend()
plt.grid(True)
plt.show()
