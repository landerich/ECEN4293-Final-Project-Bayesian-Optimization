import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# def point_logger(x_test, kernel_name, mu, std, acquisition, next_idx, run_id=None, filename="point_default.csv"):
#     """
    
#     """
#     x_test = np.asarray(x_test)
#     mu = np.asarray(mu)
#     std = np.asarray(std)
#     acquisition = np.asarray(acquisition)

#     if not (x_test.shape[0] == mu.shape[0] == std.shape[0] == acquisition.shape[0]):
#         raise ValueError("Arrays must have the same length along axis 0.")

#     m = x_test.shape[0]

#     is_selected = np.zeros(m, dtype=bool)

#     if 0 <= next_idx < m:
#         is_selected[next_idx] = True
#     else:
#         raise IndexError("next_idx is out of bounds for the point logger.")

#     data = {
#         "run_id": [run_id] * m,
#         "kernel_name": [kernel_name] * m,
#         "x_0": x_test,
#         "mu": mu,
#         "std": std,
#         "acquisition": acquisition,
#         "selected": is_selected,
#     }

#     df = pd.DataFrame(data)

#     file_exists = os.path.exists(filename)

#     df.to_csv(filename, mode='a', index=False, header=not file_exists)

#     return df

# def summary_logger(run_id, kernel_name, noise_std, kappa, next_idx, x_next, acquisition_max, filename = "summary_default.csv"):
#     """
    
#     """
#     data = {
#         "run_id": [run_id],
#         "kernel_name": [kernel_name],
#         "noise_std": [noise_std],
#         "kappa": [kappa],
#         "next_idx": [next_idx],
#         "x_next": [x_next],
#         "acquisition_max": [acquisition_max]
#     }

#     df = pd.DataFrame(data)

#     file_exists = os.path.isfile(filename)

#     df.to_csv(filename, mode = 'a', index=False, header=not file_exists)

#     return df

# def run_1d_bo_loop(objective_function, kernel_function, kernel_name,
#                    train_data_x, train_data_y, test_data, noise_std, kappa,
#                    n_iterations, **kernel_parameters):

#     train_x = np.asarray(train_data_x).copy()
#     train_y = np.asarray(train_data_y).copy()

#     results = []

#     for i in range(n_iterations):
#         result = test_bo(
#             kernel_function=kernel_function,
#             kernel_name=kernel_name,
#             train_data_x=train_x,
#             train_data_y=train_y,
#             test_data=test_data,
#             noise_std=noise_std,
#             kappa=kappa,
#             run_id=i + 1,
#             **kernel_parameters
#         )

#         x_next = result["x_next"]
#         y_next = objective_function(x_next)

#         train_x = np.append(train_x, x_next)
#         train_y = np.append(train_y, y_next)

#         results.append(result)

   
#         plot_bo(
#             test_data=test_data,
#             mu=result["mu"],
#             std=result["std"],
#             acquisition=result["acquisition"],
#             next_idx=result["next_idx"],
#             objective_function=objective_function,
#             train_x=train_x,
#             train_y=train_y
#         )

#     return results, train_x, train_y

# def plot_bo(test_data, mu, std, acquisition, next_idx, objective_function, train_x=None, train_y=None):
#     true_y = objective_function(test_data)

#     fig, ax = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

#     ax[0].plot(test_data, true_y, label="True curve: 1/x", color="black", linewidth=2)
#     ax[0].plot(test_data, mu, label="Posterior mean", color="tab:blue")
#     ax[0].fill_between(test_data, mu - 2*std, mu + 2*std, alpha=0.2, label="Uncertainty", color="tab:blue")
#     ax[0].axvline(test_data[next_idx], color="red", linestyle="--", label="Selected point")

#     if train_x is not None and train_y is not None:
#         ax[0].scatter(train_x, train_y, color="green", s=50, label="Observed points", zorder=5)

#     ax[0].set_title("Posterior vs True Curve")
#     ax[0].legend()

#     ax[1].plot(test_data, acquisition, label="Acquisition", color="tab:orange")
#     ax[1].axvline(test_data[next_idx], color="red", linestyle="--", label="Selected point")
#     ax[1].set_title("Acquisition")
#     ax[1].legend()

#     plt.tight_layout()
#     plt.show()

# results_sum, final_x_sum, final_y_sum = run_1d_bo_loop(
#     objective_function=one_over_x,
#     kernel_function=squared_exponential_kernel,
#     kernel_name="SE+Linear",
#     train_data_x=train_data_x,
#     train_data_y=train_data_y,
#     test_data=test_data,
#     noise_std=0.01,
#     kappa=2.0,
#     n_iterations=5,
#     # ell_se=1.0,
#     sigma_se=1.0,
#     length_scale=1.0
#     # sigma_linear=1.0,
# )

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

X = np.arange(0, 5, 0.25)
Y = np.arange(0, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = 1/(X**0.5 * Y**0.5) + np.sin(4*R)

surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title('3D Surface Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
fig.colorbar(surface, shrink=0.5, aspect=5)

plt.show()



# X = np.arange(0, 5, 0.25)
# Y = np.arange(0, 5, 0.25)
# X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = 1/(X**0.5 * Y**0.5) + np.sin(4*R)




# def test_function(x, y):
#     return 1.0 / (x * y)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# np.random.seed(42)

# x_data = np.linspace(0.1, 5, 20)
# y_data = np.linspace(0.1, 5, 20)
# x, y = np.meshgrid(x_data, y_data)
# z = test_function(x, y)



# noise_level = 0.0284
# x_noisy = x + np.random.normal(0, noise_level, x.shape)
# y_noisy = y + np.random.normal(0, noise_level, y.shape)
# z_noisy = z + np.random.normal(0, noise_level, z.shape)

# surface = ax.plot_surface(x_noisy, y_noisy, z_noisy, cmap='viridis', edgecolor='none')

# ax.set_title('3D Surface Plot')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# fig.colorbar(surface, shrink=0.5, aspect=5)

# plt.show()

# ============================================================================================================
# ---------------------------------------- TESTING FUNCTION --------------------------------------------------

# def test_bo(kernel_function, kernel_name, train_data_x, train_data_y, test_data, noise_std, kappa, run_id, **kernel_parameters):
#     mu, cov = gp_posterior(train_data_x, train_data_y, test_data, noise_std, kernel_function, **kernel_parameters)
#     std = posterior_std(cov)
#     acquisition = acquisition_straddle(mu, std, kappa)
#     next_idx = np.argmax(acquisition)
#     x_next = test_data[next_idx]
#     return {
#         "mu": mu, "cov": cov, "std": std, "acquisition": acquisition,
#         "next_idx": next_idx, "x_next": x_next, "run_id": run_id, "kernel_name": kernel_name
#     }

# def run_1d_bo_loop(objective_function, kernel_function, kernel_name,
#                    train_data_x, train_data_y, test_data, noise_std, kappa,
#                    n_iterations, **kernel_parameters):
#     train_x = np.atleast_2d(np.asarray(train_data_x, dtype=float)).copy()
#     train_y = np.asarray(train_data_y, dtype=float).copy()
#     results = []
#     for i in range(n_iterations):
#         result = test_bo(kernel_function, kernel_name, train_x, train_y, test_data,
#                          noise_std, kappa, i + 1, **kernel_parameters)
#         x_next = result["x_next"]
#         y_next = objective_function(x_next)
#         train_x = np.vstack([train_x, x_next])
#         train_y = np.append(train_y, y_next)
#         results.append(result)
#     return results, train_x, train_y


# def two_d_objective(x_vec, eps=1e-6):
#     x1, x2 = x_vec
#     prod = x1 * x2
#     R = (x1**2 + x2**2)
#     prod_safe = prod if abs(prod) > eps else eps
#     return min(2, 1.0 / prod_safe) # + np.sin(R)) #* np.cos(x2) Use a white noise gaussian model, check how BO works with that.

# def visualize_2d_bo(X1, X2, MU, ACQ, train_x, train_y, x_next):
#     """
#     - 3D surface of the posterior mean,
#     - scatter of all observed training points,
#     - the final chosen point highlighted,
#     - 2D contour of the acquisition function.
#     """
#     fig = plt.figure(figsize=(12, 5))

#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     surf = ax1.plot_surface(X1, X2, MU, cmap='viridis', edgecolor='none', alpha=0.8)
#     ax1.scatter(train_x[:, 0], train_x[:, 1], train_y,
#                color='red', s=40, label='Observed points')
#     ax1.scatter(x_next[0], x_next[1], two_d_objective(x_next),
#                color='black', s=70, label='Final chosen point')
#     ax1.set_title('Posterior Mean Surface (2D BO)')
#     ax1.set_xlabel('x1')
#     ax1.set_ylabel('x2')
#     ax1.set_zlabel('f(x1, x2)')
#     ax1.legend()
#     fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

#     ax2 = fig.add_subplot(1, 2, 2)
#     contour = ax2.contourf(X1, X2, ACQ, levels=30, cmap='plasma')
#     ax2.scatter(train_x[:, 0], train_x[:, 1],
#                color='white', edgecolor='black', s=40, label='Observed points')
#     ax2.scatter(x_next[0], x_next[1],
#                color='cyan', edgecolor='black', s=70, label='Final chosen point')
#     ax2.set_title('Acquisition Function (UCB)')
#     ax2.set_xlabel('x1')
#     ax2.set_ylabel('x2')
#     ax2.legend()
#     fig.colorbar(contour, ax=ax2, shrink=0.5, aspect=10)

#     plt.tight_layout()
#     plt.savefig("bo_2d_result.png", dpi=150)
#     plt.show()


# def plot_chosen_point_3d(X1, X2, objective_function, x_next, train_x=None, train_y=None):
#     """
#     Dedicated 3D plot that isolates the final chosen point on the TRUE
#     objective surface, with a vertical stem line down to the base plane
#     so its exact (x1, x2, f) location is unambiguous.

#     Args:
#         X1, X2: meshgrid arrays, shape (m, m)
#         objective_function: callable taking a (2,) vector and returning a scalar
#         x_next: shape (2,), the final selected point
#         train_x: optional (N, 2) array of all sampled training points
#         train_y: optional (N,) array of the corresponding objective values
#     """
#     Z_true = np.array([
#         objective_function(np.array([x1, x2]))
#         for x1, x2 in zip(X1.ravel(), X2.ravel())
#     ]).reshape(X1.shape)

#     z_next = objective_function(x_next)
#     z_base = Z_true.min()  # base of the stem line

#     fig = plt.figure(figsize=(8, 7))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot_surface(X1, X2, Z_true, cmap='viridis', edgecolor='none', alpha=0.55)

#     if train_x is not None and train_y is not None:
#         ax.scatter(train_x[:, 0], train_x[:, 1], train_y,
#                   color='dimgray', s=25, alpha=0.8, label='Observed points')

#     # Stem line from the base plane up to the chosen point
#     ax.plot([x_next[0], x_next[0]], [x_next[1], x_next[1]], [z_base, z_next],
#            color='red', linewidth=2, linestyle='--')

#     # The chosen point itself, marked prominently
#     ax.scatter(x_next[0], x_next[1], z_next,
#               color='red', s=120, edgecolor='black', depthshade=False,
#               label=f'Final chosen point\n(x1={x_next[0]:.3f}, x2={x_next[1]:.3f}, f={z_next:.3f})')

#     ax.set_title('Final Chosen Point on True Objective Surface')
#     ax.set_xlabel('x1')
#     ax.set_ylabel('x2')
#     ax.set_zlabel('f(x1, x2)')
#     ax.legend(loc='upper left')

#     plt.tight_layout()
#     plt.savefig("bo_2d_chosen_point_3d.png", dpi=150)
#     plt.show()


# # ==============================================================
# #                        2D BO demo driver
# # ==============================================================

# def run_2d_bo_demo(n_iterations: int = 10,
#                    noise_std: float = 0.01,
#                    kappa: float = 2.0,
#                    kernel_function=squared_exponential_kernel,
#                    kernel_name: str = "SE 2D",
#                    length: float = 1.0,
#                    sigma_se: float = 1.0):
#     """
#     Runs a full 2D BO experiment on two_d_objective and visualizes the
#     posterior surface, acquisition contour, and final chosen point.
#     """
#     x1_grid = np.linspace(0.5, 5.0, 40)  # domain kept away from 0 to avoid singularities
#     x2_grid = np.linspace(0.5, 5.0, 40)
#     X1, X2 = np.meshgrid(x1_grid, x2_grid)

#     X_test = np.column_stack([X1.ravel(), X2.ravel()])  # shape (M, 2)

#     train_x = np.array([
#         [1.0, 1.0],
#         [2.0, 3.0],
#         [4.0, 2.0],
#     ])  # shape (N, 2)

#     train_y = np.array([two_d_objective(p) for p in train_x])

#     results, final_x, final_y = run_1d_bo_loop(
#         objective_function=two_d_objective,
#         kernel_function=kernel_function,
#         kernel_name=kernel_name,
#         train_data_x=train_x,
#         train_data_y=train_y,
#         test_data=X_test,
#         noise_std=noise_std,
#         kappa=kappa,
#         n_iterations=n_iterations,
#         length=length,
#         sigma_se=sigma_se
#     )

#     last = results[-1]
#     mu = last["mu"]
#     acquisition = last["acquisition"]
#     x_next = last["x_next"]

#     # FIX: use .shape, not .reshape (which is a bound method, not a tuple)
#     MU = mu.reshape(X1.shape)
#     ACQ = acquisition.reshape(X1.shape)

#     visualize_2d_bo(
#         X1, X2,
#         MU,
#         ACQ,
#         train_x=final_x,
#         train_y=final_y,
#         x_next=x_next
#     )

#     # Dedicated 3D view of the final chosen point on the true surface
#     plot_chosen_point_3d(
#         X1, X2,
#         objective_function=two_d_objective,
#         x_next=x_next,
#         train_x=final_x,
#         train_y=final_y
#     )

#     print(f"Final chosen point: x1 = {x_next[0]:.4f}, x2 = {x_next[1]:.4f}")
#     print(f"Objective value at final point: {two_d_objective(x_next):.4f}")

#     return results, final_x, final_y, x_next