from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.signal import find_peaks

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


def separatrixes(hamilton_2D: NumpyArray):
    y_loc = hamilton_2D[hamilton_2D.shape[0] // 2, :]

    peaks, _ = find_peaks(y_loc)

    h_levels = np.array([y_loc[peak] for peak in peaks])
    return h_levels


def calc_hamiltonian(
    p_turn0: NumpyArray,
    p_turn1: NumpyArray,
    x_turn0: NumpyArray,
    x_turn1: NumpyArray,
    atol: float = 1e-5,
    maxiter: int = 1000,
):
    """
    Calculate Hamiltonian from set of points in turn 0 and turn 1

    Parameters
    ----------
    p_turn0
        Y positions in turn 0
    p_turn1
        Y positions in turn 1
    x_turn0
        X positions in turn 0
    x_turn1
        X positions in turn 1
    atol
        Absolute tolerance to
    maxiter
        Maximum number of refinement passes to reach atol

    Returns
    -------
    hamilton_2D
        2D Hamiltonian
    """
    warnings.warn("Using wrong hamiltonian calculation, solver needs to be rewritten!")

    scale_x = np.max(np.abs(x_turn0))
    scale_y = np.max(np.abs(p_turn0))

    dx = x_turn1[:, :] - x_turn0[:, :]
    dp = p_turn1[:, :] - p_turn0[:, :]
    # Compute derivatives of Hamiltonian
    dH_dp_grid = dx
    dH_dx_grid = -dp
    X, P = x_turn0, p_turn0
    xi = x_turn0[0, :]
    pi = p_turn0[:, 0]
    # ---- Integrate to reconstruct Hamiltonian ----
    # We integrate along p (rows), then x (columns)
    hamilton_2D = np.zeros_like(X)
    motion = np.sqrt(dx**2 + dp**2)
    # Find index of the smallest motion: this is the best estimate of the fixed point
    fixed_idx = np.unravel_index(np.argmin(motion.flatten()), motion.shape)
    # Integrate along p for each x (column-wise integration)
    for iteration in range(maxiter):
        for i in range(hamilton_2D.shape[1]):
            if np.all(np.isnan(dH_dp_grid[:, i])):
                continue
            hamilton_2D[:, i] += cumtrapz(dH_dp_grid[:, i], pi, initial=0)
        for j in range(hamilton_2D.shape[0]):
            if np.all(np.isnan(dH_dx_grid[j, :])):
                continue
            hamilton_2D[j, :] += cumtrapz(dH_dx_grid[j, :], xi, initial=0)
        PLOT = True
        if PLOT:
            plt.figure(10)
            plt.clf()
            plt.matshow(
                np.gradient(hamilton_2D, pi, axis=0, edge_order=2) - dH_dp_grid,
                fignum=0,
            )
            plt.colorbar()
        err1 = np.gradient(hamilton_2D, pi, axis=0, edge_order=2) - dH_dp_grid
        dH_dp_grid = -0.1 * err1
        if PLOT:
            plt.figure(11)
            plt.clf()
            plt.matshow(
                np.gradient(hamilton_2D, xi, axis=1, edge_order=2) - dH_dx_grid,
                fignum=0,
            )
            plt.colorbar()
        err2 = np.gradient(hamilton_2D, xi, axis=1, edge_order=2) - dH_dx_grid
        dH_dx_grid = -0.1 * err2
        if PLOT:
            plt.figure(12)
            plt.clf()
            plt.matshow(hamilton_2D, fignum=0)
            plt.draw()
            plt.pause(0.1)
        # normalize error by scale of grid.
        # otherwise the error is absolute
        err1 /= scale_x
        err2 /= scale_y
        if (np.max(np.abs(err1)) < atol) and (np.max(np.abs(err2)) < atol):
            break
    return hamilton_2D
