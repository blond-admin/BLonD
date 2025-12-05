from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def calc_ellipse_gamma(alpha: float, beta: float) -> float:
    gamma = (1 + alpha**2) / beta
    return gamma


def ellipse_residuals(
    x: NumpyArray,
    y: NumpyArray,
    alpha: float,
    beta: float,
    epsilon: float,
) -> NumpyArray:
    """Calculates how

    See https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters

    Parameters
    ----------
    x
    y
    alpha
    beta
    epsilon

    Returns
    -------

    """
    gamma = calc_ellipse_gamma(alpha, beta)
    epsilon2 = gamma * x**2 + 2 * alpha * x * y + beta * y**2
    return np.abs(epsilon2 - epsilon)


def _ellipse_residuals_helper(
    params: tuple[float, float, float],
    x: NumpyArray,
    y: NumpyArray,
):
    return ellipse_residuals(
        x,
        y,
        alpha=params[0],
        beta=params[1],
        epsilon=params[2],
    )


def fit_ellipse(
    x: NumpyArray,
    y: NumpyArray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    p0 = [0, 1, 1]
    x = x / scale_x
    y = y / scale_y
    sol = least_squares(
        _ellipse_residuals_helper,
        p0,
        args=(x, y),
    )
    alpha, beta, epsilon = sol.x
    ret = (
        float(alpha),
        float(beta * scale_x / scale_y),
        float(epsilon * scale_x * scale_y),
    )
    return ret


def plot_ellipse(
    alpha: float,
    beta: float,
    epsilon: float,
    n_points: int = 400,
    ax: matplotlib.axes.Axes | None = None,
):
    """Plot the Courant–Snyder (Twiss) ellipse.

    Parameters
    ----------
    alpha
    beta
    epsilon
    n_points
    ax
        Axes on which to draw. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # parameter angle
    theta = np.linspace(0, 2 * np.pi, n_points)

    # parametric form of the ellipse in Courant–Snyder variables
    x = np.sqrt(beta * epsilon) * np.cos(theta)
    y = -alpha * np.sqrt(epsilon / beta) * np.cos(theta) + np.sqrt(
        epsilon / beta
    ) * np.sin(theta)

    if ax is None:
        ax = plt.gca()

    ax.plot(x, y, label=f"α={alpha}, β={beta}, ε={epsilon}")

    return ax
