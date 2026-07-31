import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os

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


def squared_exponential_kernel(x1: float, x2: float, length_scale: float = 1.0, sigma_se: float = 1.0) -> float:
    """ 
    Kernel function: Covariance function that returns the scalar covariance value.
    ----------
    Args:
        x1: Scalar input 1
        x2: Scalar input 2
        length_scale: 
        sigma_se: signal variance / amplitude

    Returns:
        float: Covariance between x1 and x2 under the squared sum exponential kernel.

    """
    cov = (x1 - x2) ** 2
    lgt = 2*(length_scale**2)
    
    return sigma_se**2 * np.exp(-(cov/lgt))

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
    arr1 = np.atleast_1d(arr1)
    arr2 = np.atleast_1d(arr2)

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

def combined_kernel_sum(xin1, xin2, ell_se, sigma_se, sigma_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sigma_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sigma_linear)
    return k_se + k_linear

def combined_kernel_product(xin1, xin2, ell_se, sigma_se, sigma_linear):
    k_se = squared_exponential_kernel(x1= xin1, x2= xin2, length_scale= ell_se, sigma_se= sigma_se)
    k_linear = linear_kernel(x1= xin1, x2= xin2, sigma_linear= sigma_linear)
    return k_se * k_linear

# mu_s, cov_post = gp_posterior(X_train=X_train_two, y_train=y_train_two, X_test=X_test_two, ell=ell_two, sigma=sigma_two, noise_std=noise_std_two)
# std_s = posterior_std(cov_post)
# ucb_vals = acquisition_ucb(mu_s, std_s, kappa_two)
# next_idx = [np.argmax(ucb_vals)]

# print(f"\n\t------------------------------")
# print(f"MU_S:\t{mu_s}\n COV_POST:\t{cov_post}")
# print(f"\nSTD:\t{std_s}")
# print(f"\nUBC VALUES:\t{ucb_vals}")
# print(f"\nNEST_IDX:\t{next_idx}")
# print(f"\n\t------------------------------")

# =================================================================
#  Test over 1/x function for kernel switch on asymptotic behavior
# =================================================================

def one_over_x(x):
    return 1/x

train_data_x = np.array([
0.5,
0.75,
1.0,
1.25,
1.5,
1.75,
2.0,
2.25,
2.5,
2.75,
3.0,
3.25,
3.5,
3.75,
4.0,
])
train_data_y = one_over_x(train_data_x)

test_data = np.linspace(0.001, 6.0, 1000)

# =================================================================
#                        Testing function
# =================================================================

def test_bo(kernel_function, kernel_name, train_data_x, train_data_y, test_data, noise_std, kappa, run_id, **kernel_parameters):
    # Compute posterior
    mu, cov = gp_posterior(X_train=train_data_x,
                           y_train=train_data_y,
                           X_test=test_data,
                           noise_std=noise_std,
                           kernel_function=kernel_function,
                           **kernel_parameters)
    
    # Compute std 
    standard_deviation = posterior_std(cov)

    # Compute acquisition
    acquisition = acquisition_ucb(mu=mu,
                                  std= standard_deviation,
                                  kappa=kappa)

    # compute selected next point
    next_idx = np.argmax(acquisition)
    x_next = test_data[next_idx]
    # return point-level table, a summary dictionary 


    return {
        "mu": mu,
        "cov": cov,
        "std": standard_deviation,
        "acquisition": acquisition,
        "next_idx": next_idx,
        "x_next": x_next,
        "run_id": run_id,
        "kernel_name": kernel_name
    }

def run_1d_bo_loop(objective_function, kernel_function, kernel_name,
                   train_data_x, train_data_y, test_data, noise_std, kappa,
                   n_iterations, **kernel_parameters):

    train_x = np.asarray(train_data_x).copy()
    train_y = np.asarray(train_data_y).copy()

    results = []

    for i in range(n_iterations):
        result = test_bo(
            kernel_function=kernel_function,
            kernel_name=kernel_name,
            train_data_x=train_x,
            train_data_y=train_y,
            test_data=test_data,
            noise_std=noise_std,
            kappa=kappa,
            run_id=i + 1,
            **kernel_parameters
        )

        x_next = result["x_next"]
        y_next = objective_function(x_next)

        train_x = np.append(train_x, x_next)
        train_y = np.append(train_y, y_next)

        results.append(result)

   
        plot_bo(
            test_data=test_data,
            mu=result["mu"],
            std=result["std"],
            acquisition=result["acquisition"],
            next_idx=result["next_idx"],
            objective_function=objective_function,
            train_x=train_x,
            train_y=train_y
        )

    return results, train_x, train_y

def plot_bo(test_data, mu, std, acquisition, next_idx, objective_function, train_x=None, train_y=None):
    true_y = objective_function(test_data)

    fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

    ax[0].plot(test_data, true_y, label="True curve: 1/x", color="black", linewidth=2)
    ax[0].plot(test_data, mu, label="Posterior mean", color="tab:blue")
    ax[0].fill_between(test_data, mu - 2*std, mu + 2*std, alpha=0.2, label="Uncertainty", color="tab:blue")
    ax[0].axvline(test_data[next_idx], color="red", linestyle="--", label="Selected point")

    if train_x is not None and train_y is not None:
        ax[0].scatter(train_x, train_y, color="green", s=50, label="Observed points", zorder=5)

    ax[0].set_title("Posterior vs True Curve")
    ax[0].legend()

    ax[1].plot(test_data, acquisition, label="Acquisition", color="tab:orange")
    ax[1].axvline(test_data[next_idx], color="red", linestyle="--", label="Selected point")
    ax[1].set_title("Acquisition")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def point_logger(x_test, kernel_name, mu, std, acquisition, next_idx, run_id=None, filename="point_default.csv"):
    """
    
    """
    x_test = np.asarray(x_test)
    mu = np.asarray(mu)
    std = np.asarray(std)
    acquisition = np.asarray(acquisition)

    if not (x_test.shape[0] == mu.shape[0] == std.shape[0] == acquisition.shape[0]):
        raise ValueError("Arrays must have the same length along axis 0.")

    m = x_test.shape[0]

    is_selected = np.zeros(m, dtype=bool)

    if 0 <= next_idx < m:
        is_selected[next_idx] = True
    else:
        raise IndexError("next_idx is out of bounds for the point logger.")

    data = {
        "run_id": [run_id] * m,
        "kernel_name": [kernel_name] * m,
        "x_0": x_test,
        "mu": mu,
        "std": std,
        "acquisition": acquisition,
        "selected": is_selected,
    }

    df = pd.DataFrame(data)

    file_exists = os.path.exists(filename)

    df.to_csv(filename, mode='a', index=False, header=not file_exists)

    return df

def summary_logger(run_id, kernel_name, noise_std, kappa, next_idx, x_next, acquisition_max, filename = "summary_default.csv"):
    """
    
    """
    data = {
        "run_id": [run_id],
        "kernel_name": [kernel_name],
        "noise_std": [noise_std],
        "kappa": [kappa],
        "next_idx": [next_idx],
        "x_next": [x_next],
        "acquisition_max": [acquisition_max]
    }

    df = pd.DataFrame(data)

    file_exists = os.path.isfile(filename)

    df.to_csv(filename, mode = 'a', index=False, header=not file_exists)

    return df

# results_se, final_x_se, final_y_se = run_1d_bo_loop(
#     objective_function=one_over_x,
#     kernel_function=squared_exponential_kernel,
#     kernel_name="Square Exponential",
#     train_data_x=np.array([1.0, 2.0, 4.0]),
#     train_data_y=one_over_x(np.array([1.0, 2.0, 4.0])),
#     test_data=np.linspace(0.001, 5.0, 200),
#     noise_std=0.01,
#     kappa=2.0,
#     n_iterations=5,
#     length_scale=1.0,
#     sigma=1.0,
# )

# results_lin, final_x_lin, final_y_lin = run_1d_bo_loop(
#     objective_function=one_over_x,
#     kernel_function=linear_kernel,
#     kernel_name="Linear",
#     train_data_x=...,
#     train_data_y=...,
#     test_data=...,
#     noise_std=0.01,
#     kappa=2.0,
#     n_iterations=5,
#     sigma_linear=1.0,
# )

results_sum, final_x_sum, final_y_sum = run_1d_bo_loop(
    objective_function=one_over_x,
    kernel_function=squared_exponential_kernel,
    kernel_name="SE+Linear",
    train_data_x=train_data_x,
    train_data_y=train_data_y,
    test_data=test_data,
    noise_std=0.01,
    kappa=2.0,
    n_iterations=5,
    # ell_se=1.0,
    sigma_se=1.0,
    length_scale=1.0
    # sigma_linear=1.0,
)

def edge_case():    # What is a safe measurable value (i.e., how high off in the y axis is acceptable and usable in practice?)

    veredict = None
    
    return veredict

# If f(x) > some value, clamp it to a safe measurable value.
# f(x, n=1) = n if 1/x > n else 1/x

# Jul 27th 2026
# Look for the best combination of c2q that is under a certain threshold
# Slice Z axis at a chosen safe threshold
# Start from the corner of the plane and move towards the slope until the c2q latches
# Extend the code for 2D usage Quadrant 1 of 1/xy

# Goal here is to record: How many times it gets called
# How close is the BO to the actual function.