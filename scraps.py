import os
import pandas as pd
import numpy as np

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