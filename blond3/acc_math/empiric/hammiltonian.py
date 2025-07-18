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
    if len(h_levels) == 0: # fallback options if y_loc has no peaks
        h_levels = [np.max(y_loc)]
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
    DEV_PLOT = False

    # User must provide:
    # ---- Compute delta values ----
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
    H = np.zeros_like(X)
    # Integrate along p for each x (column-wise integration)
    for iteration in range(maxiter):
        for i in range(H.shape[1]):
            if np.all(np.isnan(dH_dp_grid[:, i])):
                continue
            H[:, i] += cumtrapz(dH_dp_grid[:, i], pi, initial=0)
        for j in range(H.shape[0]):
            if np.all(np.isnan(dH_dx_grid[j, :])):
                continue
            H[j, :] += cumtrapz(dH_dx_grid[j, :], xi, initial=0)
        if DEV_PLOT:
            plt.figure(10)
            plt.clf()
            plt.matshow(np.gradient(H, pi, axis=0, edge_order=2) - dH_dp_grid, fignum=0)
            plt.colorbar()
        err1 = np.gradient(H, pi, axis=0, edge_order=2) - dH_dp_grid
        dH_dp_grid = -0.1 * err1
        if DEV_PLOT:
            plt.figure(11)
            plt.clf()
            plt.matshow(np.gradient(H, pi, axis=1, edge_order=2) - dH_dx_grid, fignum=0)
            plt.colorbar()
        err2 = np.gradient(H, xi, axis=1, edge_order=2) - dH_dx_grid
        dH_dx_grid = -0.1 * err2
        if DEV_PLOT:
            plt.figure(12)
            plt.clf()
            plt.matshow(H, fignum=0)
            plt.colorbar()

            plt.draw()
            plt.pause(0.1)
        H -= np.min(H)
        H /= np.max(H)
        err1_rel = err1.max() * (pi[1] - pi[0])
        err2_rel = err2.max() * (xi[1] - xi[0])
        print(atol)
        print(err1_rel, err2_rel)
        print((err1_rel < atol) and (err2_rel < atol))
        if (err1_rel < atol) and (err2_rel < atol) and iteration > 1:
            print(f"breaking with {(err1_rel, err2_rel)}")
            break
    return H
