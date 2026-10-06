"""Visualization helpers for plotting two-variable functions in 3D."""

import matplotlib.pyplot as plt
import numpy as np
import scipy


def plot_3d_function(
	function,
	x_bounds=(-5.0, 5.0),
	y_bounds=(-5.0, 5.0),
	resolution=100,
	title=None,
	ax=None,
	show=True,
):
	"""Plot ``function(x, y)`` as a 3D surface.

	``function`` may be a NumPy/SciPy ufunc or any callable that accepts
	NumPy arrays. Returns the figure and axes so callers can customize them.
	"""
	if resolution < 2:
		raise ValueError("resolution must be at least 2")
	if len(x_bounds) != 2 or len(y_bounds) != 2:
		raise ValueError("bounds must contain exactly two values")

	x = np.linspace(*x_bounds, resolution)
	y = np.linspace(*y_bounds, resolution)
	X, Y = np.meshgrid(x, y)
	Z = np.asarray(function(X, Y))

	if Z.shape != X.shape:
		raise ValueError("function must return an array with the input shape")

	if ax is None:
		figure = plt.figure(figsize=(10, 7))
		ax = figure.add_subplot(111, projection="3d")
	else:
		figure = ax.figure

	surface = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_zlabel("f(x, y)")
	ax.set_title(title or getattr(function, "__name__", "Function"))
	figure.colorbar(surface, ax=ax, shrink=0.6, pad=0.1, label="f(x, y)")

	if show:
		plt.show()
	return figure, ax


__all__ = ["plot_3d_function"]
