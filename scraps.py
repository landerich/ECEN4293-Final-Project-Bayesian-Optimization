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