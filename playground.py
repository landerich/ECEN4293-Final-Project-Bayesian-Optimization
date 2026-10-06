import numpy as np
import scipy as sp
# import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =================================================================
# ====================== Testing Values ===========================
# =================================================================
#  Test over 1/x function for kernel switch on asymptotic behavior

rng = np.random.default_rng()

def test_function(x, y):
    return 1/(x*y)

np.random.seed(42)
x_data = np.linspace(0, 5, 20)
y_data = np.linspace(0, 5, 20)
x, y = np.meshgrid(x_data, y_data)
z = test_function(x_data, y_data)



noise_level = 0.0284
x_noisy = x + np.random.normal(0, noise_level, x.shape)
y_noisy = y + np.random.normal(0, noise_level, y.shape)
z_noisy = z + np.random.normal(0, noise_level, z.shape)


# =================================================================
# =================================================================

np.set_printoptions(precision=2, suppress=True)

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

def linear_kernel(x1:np.ndarray,
                  x2:np.ndarray,
                  sigma_linear:float = 1.0) -> float:
    """
    
    """
    return sigma_linear**2 * np.dot(x1, x2)

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
    return np.sqrt(diag)

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

def acquisition_straddle(mu, std, threshold, beta=1.96):
    mu_straddle = np.asarray(mu)
    std_straddle = np.asarray(std)

    if mu_straddle.shape != std_straddle.shape:
        raise ValueError("Mu and STD vectors must be the same size to compute straddle.")

    return beta * std_straddle - np.abs(mu_straddle - threshold)



# =================================================================
#                        Testing function
# =================================================================

def test_bo(kernel_function, kernel_name, train_data_x, train_data_y, test_data, noise_std, threshold, run_id, **kernel_parameters):
    mu, cov = gp_posterior(train_data_x, train_data_y, test_data, noise_std, kernel_function, **kernel_parameters)
    std = posterior_std(cov)
    acquisition = acquisition_straddle(mu, std, threshold)
    next_idx = np.argmax(acquisition)
    x_next = test_data[next_idx]
    return {
        "mu": mu, "cov": cov, "std": std, "acquisition": acquisition,
        "next_idx": next_idx, "x_next": x_next, "run_id": run_id, "kernel_name": kernel_name
    }

def run_1d_bo_loop(objective_function, kernel_function, kernel_name,
                   train_data_x, train_data_y, test_data, noise_std, kappa,
                   n_iterations, **kernel_parameters):
    train_x = np.atleast_2d(np.asarray(train_data_x, dtype=float)).copy()
    train_y = np.asarray(train_data_y, dtype=float).copy()
    results = []
    for i in range(n_iterations):
        result = test_bo(kernel_function, kernel_name, train_x, train_y, test_data,
                         noise_std, kappa, i + 1, **kernel_parameters)
        x_next = result["x_next"]
        y_next = objective_function(x_next)
        train_x = np.vstack([train_x, x_next])
        train_y = np.append(train_y, y_next)
        results.append(result)
    return results, train_x, train_y


def two_d_objective(x_vec, eps=1e-6):
    x1, x2 = x_vec
    prod = x1 * x2
    R = (x1**2 + x2**2)
    prod_safe = prod if abs(prod) > eps else eps
    return min(2, 1.0 / prod_safe) # + np.sin(R)) #* np.cos(x2) Use a white noise gaussian model, check how BO works with that.

def visualize_2d_bo(X1, X2, MU, ACQ, train_x, train_y, x_next):
    """
    - 3D surface of the posterior mean,
    - scatter of all observed training points,
    - the final chosen point highlighted,
    - 2D contour of the acquisition function.
    """
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X1, X2, MU, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.scatter(train_x[:, 0], train_x[:, 1], train_y,
               color='red', s=40, label='Observed points')
    ax1.scatter(x_next[0], x_next[1], two_d_objective(x_next),
               color='black', s=70, label='Final chosen point')
    ax1.set_title('Posterior Mean Surface (2D BO)')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x1, x2)')
    ax1.legend()
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X1, X2, ACQ, levels=30, cmap='plasma')
    ax2.scatter(train_x[:, 0], train_x[:, 1],
               color='white', edgecolor='black', s=40, label='Observed points')
    ax2.scatter(x_next[0], x_next[1],
               color='cyan', edgecolor='black', s=70, label='Final chosen point')
    ax2.set_title('Acquisition Function (UCB)')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.legend()
    fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig("bo_2d_result.png", dpi=150)
    plt.show()


def plot_chosen_point_3d(X1, X2, objective_function, x_next, train_x=None, train_y=None):
    """
    Dedicated 3D plot that isolates the final chosen point on the TRUE
    objective surface, with a vertical stem line down to the base plane
    so its exact (x1, x2, f) location is unambiguous.

    Args:
        X1, X2: meshgrid arrays, shape (m, m)
        objective_function: callable taking a (2,) vector and returning a scalar
        x_next: shape (2,), the final selected point
        train_x: optional (N, 2) array of all sampled training points
        train_y: optional (N,) array of the corresponding objective values
    """
    Z_true = np.array([
        objective_function(np.array([x1, x2]))
        for x1, x2 in zip(X1.ravel(), X2.ravel())
    ]).reshape(X1.shape)

    z_next = objective_function(x_next)
    z_base = Z_true.min()  # base of the stem line

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X1, X2, Z_true, cmap='viridis', edgecolor='none', alpha=0.55)

    if train_x is not None and train_y is not None:
        ax.scatter(train_x[:, 0], train_x[:, 1], train_y,
                  color='dimgray', s=25, alpha=0.8, label='Observed points')

    # Stem line from the base plane up to the chosen point
    ax.plot([x_next[0], x_next[0]], [x_next[1], x_next[1]], [z_base, z_next],
           color='red', linewidth=2, linestyle='--')

    # The chosen point itself, marked prominently
    ax.scatter(x_next[0], x_next[1], z_next,
              color='red', s=120, edgecolor='black', depthshade=False,
              label=f'Final chosen point\n(x1={x_next[0]:.3f}, x2={x_next[1]:.3f}, f={z_next:.3f})')

    ax.set_title('Final Chosen Point on True Objective Surface')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig("bo_2d_chosen_point_3d.png", dpi=150)
    plt.show()


# ==============================================================
#                        2D BO demo driver
# ==============================================================

def run_2d_bo_demo(n_iterations: int = 10,
                   noise_std: float = 0.01,
                   kappa: float = 2.0,
                   kernel_function=squared_exponential_kernel,
                   kernel_name: str = "SE 2D",
                   length: float = 1.0,
                   sigma_se: float = 1.0):
    """
    Runs a full 2D BO experiment on two_d_objective and visualizes the
    posterior surface, acquisition contour, and final chosen point.
    """
    x1_grid = np.linspace(0.5, 5.0, 40)  # domain kept away from 0 to avoid singularities
    x2_grid = np.linspace(0.5, 5.0, 40)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    X_test = np.column_stack([X1.ravel(), X2.ravel()])  # shape (M, 2)

    train_x = np.array([
        [1.0, 1.0],
        [2.0, 3.0],
        [4.0, 2.0],
    ])  # shape (N, 2)

    train_y = np.array([two_d_objective(p) for p in train_x])

    results, final_x, final_y = run_1d_bo_loop(
        objective_function=two_d_objective,
        kernel_function=kernel_function,
        kernel_name=kernel_name,
        train_data_x=train_x,
        train_data_y=train_y,
        test_data=X_test,
        noise_std=noise_std,
        kappa=kappa,
        n_iterations=n_iterations,
        length=length,
        sigma_se=sigma_se
    )

    last = results[-1]
    mu = last["mu"]
    acquisition = last["acquisition"]
    x_next = last["x_next"]

    # FIX: use .shape, not .reshape (which is a bound method, not a tuple)
    MU = mu.reshape(X1.shape)
    ACQ = acquisition.reshape(X1.shape)

    visualize_2d_bo(
        X1, X2,
        MU,
        ACQ,
        train_x=final_x,
        train_y=final_y,
        x_next=x_next
    )

    # Dedicated 3D view of the final chosen point on the true surface
    plot_chosen_point_3d(
        X1, X2,
        objective_function=two_d_objective,
        x_next=x_next,
        train_x=final_x,
        train_y=final_y
    )

    print(f"Final chosen point: x1 = {x_next[0]:.4f}, x2 = {x_next[1]:.4f}")
    print(f"Objective value at final point: {two_d_objective(x_next):.4f}")

    return results, final_x, final_y, x_next


if __name__ == "__main__":
    run_2d_bo_demo(n_iterations=10, kernel_function=squared_exponential_kernel,
                   kernel_name="SE 2D", length=1.0, sigma_se=1.0)

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