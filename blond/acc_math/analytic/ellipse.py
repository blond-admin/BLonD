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


def get_points_on_ellipse(alpha, beta, epsilon, n_points):
    # parameter angle
    theta = np.linspace(0, 2 * np.pi, n_points)
    # parametric form of the ellipse in Courant–Snyder variables
    x = np.sqrt(beta * epsilon) * np.cos(theta)
    y = -alpha * np.sqrt(epsilon / beta) * np.cos(theta) + np.sqrt(
        epsilon / beta
    ) * np.sin(theta)
    return x, y


def plot_ellipse(
    alpha: float,
    beta: float,
    epsilon: float,
    n_points: int = 400,
    ax: matplotlib.axes.Axes | None = None,
    **kwargs_plot,
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
    x, y = get_points_on_ellipse(alpha, beta, epsilon, n_points)

    if ax is None:
        ax = plt.gca()

    ax.plot(x, y, label=f"α={alpha}, β={beta}, ε={epsilon}", **kwargs_plot)

    return ax


def transform_twiss(x, xp, alpha1, beta1, eps1, alpha2, beta2, eps2):
    A1 = np.sqrt(eps1) * np.array(
        [[np.sqrt(beta1), 0], [-alpha1 / np.sqrt(beta1), 1 / np.sqrt(beta1)]]
    )

    A2 = np.sqrt(eps2) * np.array(
        [[np.sqrt(beta2), 0], [-alpha2 / np.sqrt(beta2), 1 / np.sqrt(beta2)]]
    )

    M = A2 @ np.linalg.inv(A1)

    x2, xp2 = M @ np.vstack([x, xp])

    return x2, xp2




def twiss_from_cloud(x, xp):
    # compute second moments
    x2 = np.mean(x**2)
    xp2 = np.mean(xp**2)
    xxp = np.mean(x * xp)

    # rms emittance
    eps = np.sqrt(x2 * xp2 - xxp**2)

    # Twiss parameters
    beta = x2 / eps
    alpha = -xxp / eps
    gamma = xp2 / eps

    return beta, alpha, gamma, eps
