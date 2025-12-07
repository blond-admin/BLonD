# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Analytical tools for working with ellipses.

This module provides utilities for fitting, transforming, and visualizing
ellipses described by Courant-Snyder (Twiss) parameters, commonly used in
accelerator physics to characterize beam distributions in phase space.


References
----------
.. [1] https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters


Authors
-------
S. Lauber
L. Thiele
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from numpy.typing import NDArray as NumpyArray


def calc_ellipse_gamma(alpha: float, beta: float) -> float:
    """Calculate the gamma Courant-Snyder parameter from alpha and beta.

    The three Courant-Snyder parameters (alpha, beta, gamma) satisfy the relation
    ``gamma = (1 + alpha²) / beta``. This function computes gamma from the other two.

    Parameters
    ----------
    alpha
        Courant-Snyder alpha parameter (dimensionless), describing the correlation
        between position and momentum.
    beta
        Courant-Snyder beta parameter, describing the ratio of position spread to
        momentum spread. Units depend on the coordinates used.

    Returns
    -------
    float
        The gamma Courant-Snyder parameter, with units inverse to beta.

    See Also
    --------
    fit_ellipse : Fit ellipse to data points
    ellipse_residuals : Compute residuals for ellipse fitting

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters
    """
    gamma = (1 + alpha**2) / beta
    return gamma


def ellipse_residuals(
    x: NumpyArray,
    y: NumpyArray,
    alpha: float,
    beta: float,
    epsilon: float,
) -> NumpyArray:
    """Calculate residuals of points from a Courant-Snyder ellipse.

    For each point (x, y) in phase space, this function computes how far the point
    deviates from lying exactly on the ellipse defined by the Courant-Snyder parameters.
    The ellipse equation is: ``gamma·x² + 2·alpha·x·y + beta·y² = epsilon``.

    Parameters
    ----------
    x
        Array of x-coordinates (e.g., position or time in phase space).
    y
        Array of y-coordinates (e.g., momentum or energy in phase space).
        Must have the same shape as x.
    alpha
        Courant-Snyder alpha parameter (dimensionless).
    beta
        Courant-Snyder beta parameter. Units depend on x and y coordinates.
    epsilon
        Emittance, representing the area enclosed by the ellipse in phase space.

    Returns
    -------
    residuals
        Array of absolute residuals for each point, showing the deviation from
        the ellipse. Zero residual means the point lies exactly on the ellipse.

    Notes
    -----
    This function is typically used as the objective function for fitting an ellipse
    to phase space data via least-squares optimization.

    See Also
    --------
    fit_ellipse : Fit Courant-Snyder parameters to data points.
    calc_ellipse_gamma : Compute gamma from alpha and beta.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Courant%E2%80%93Snyder_parameters
    """
    gamma = calc_ellipse_gamma(alpha, beta)
    epsilon2 = gamma * x**2 + 2 * alpha * x * y + beta * y**2
    return np.abs(epsilon2 - epsilon)


def _ellipse_residuals_helper(
    params: tuple[float, float, float],
    x: NumpyArray,
    y: NumpyArray,
) -> NumpyArray:
    """Helper function for scipy.optimize.least_squares to fit ellipse parameters.

    This is an internal wrapper that unpacks the parameter tuple for use with
    scipy's least_squares optimizer.

    Parameters
    ----------
    params
        Tuple of (alpha, beta, epsilon) Courant-Snyder parameters to evaluate.
    x
        Array of x-coordinates in phase space.
    y
        Array of y-coordinates in phase space.

    Returns
    -------
    NumpyArray
        Residuals from the ellipse for the given parameters.

    See Also
    --------
    ellipse_residuals : The underlying residual function
    fit_ellipse : Uses this helper for optimization
    """
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
) -> tuple[float, float, float]:
    """Fit Courant-Snyder ellipse parameters to data points.

    This function uses least-squares optimization to find the best-fit Courant-Snyder
    parameters (alpha, beta, epsilon) that describe an ellipse containing the given
    points. The fitting is performed in scaled coordinates to improve
    numerical stability, then the results are transformed back to the original units.

    Parameters
    ----------
    x
        Array of x-coordinates (e.g., position or time in phase space).
    y
        Array of y-coordinates (e.g., momentum or energy in phase space).
        Must have the same length as x.
    scale_x
        Scaling factor for x-coordinates. The data is normalized by this value
        during fitting to improve numerical conditioning. Default is 1.0 (no scaling).
    scale_y
        Scaling factor for y-coordinates. The data is normalized by this value
        during fitting to improve numerical conditioning. Default is 1.0 (no scaling).

    Returns
    -------
    alpha : float
        Fitted Courant-Snyder alpha parameter (dimensionless).
    beta : float
        Fitted Courant-Snyder beta parameter, in units of ``[x_units²/y_units]``.
    epsilon : float
        Fitted emittance (area of ellipse), in units of ``[x_units·y_units]``.

    Notes
    -----
    - Scaling is recommended when x and y have very different magnitudes, as it
      improves the conditioning of the least-squares problem.
    - The fitted parameters are automatically transformed back to account for the
      scaling: beta is scaled by ``scale_x/scale_y`` and epsilon by ``scale_x·scale_y``.

    Examples
    --------
    Fit an ellipse to particle tracking data:

    >>> import numpy as np
    >>> from blond.acc_math.analytic.ellipse import fit_ellipse
    >>>
    >>> # Simulated phase space data
    >>> theta = np.linspace(0, 2 * np.pi, 10)
    >>> x = np.sin(theta) * 1e-9  # positions in nanoseconds
    >>> y = np.cos(theta) * 1e6   # energies in eV
    >>>
    >>> # Fit with appropriate scaling
    >>> alpha, beta, epsilon = fit_ellipse(
    ...     x, y,
    ...     scale_x=1e-9,
    ...     scale_y=1e6
    ... )
    >>> print(f"Fitted parameters: α={alpha:.3f}, β={beta:.3e}, ε={epsilon:.3e}")

    See Also
    --------
    ellipse_residuals : Residual function used for optimization
    get_points_on_ellipse : Generate points on the fitted ellipse
    plot_ellipse : Visualize the fitted ellipse
    """
    xmax = np.max(np.abs(x))
    ymax = np.max(np.abs(y))
    epsilon = xmax * ymax

    p0 = [0, xmax**2 / epsilon, epsilon]
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


def get_points_on_ellipse(
    alpha: float,
    beta: float,
    epsilon: float,
    n_points: int,
) -> tuple[NumpyArray, NumpyArray]:
    """Generate points on an ellipse for visualization.

    Computes coordinates of points uniformly distributed along an
    ellipse using the parametric representation in terms of the parameter angle theta.

    Parameters
    ----------
    alpha
        Courant-Snyder alpha parameter (dimensionless).
    beta
        Courant-Snyder beta parameter. Units determine the units of output x.
    epsilon
        Emittance parameter. Together with beta, determines the size of the ellipse.
    n_points
        Number of points to generate along the ellipse perimeter.

    Returns
    -------
    x
        Array of x-coordinates (position) on the ellipse, with shape (n_points,).
    y
        Array of y-coordinates (momentum) on the ellipse, with shape (n_points,).

    Notes
    -----
    The parametric equations used are:

    - ``x(θ) = √(β·ε) · cos(θ)``
    - ``y(θ) = -α·√(ε/β)·cos(θ) + √(ε/β)·sin(θ)``

    where θ ranges from 0 to 2π.

    Examples
    --------
    Generate and plot an ellipse:

    >>> import matplotlib.pyplot as plt
    >>> from blond.acc_math.analytic.ellipse import get_points_on_ellipse
    >>>
    >>> x, y = get_points_on_ellipse(alpha=0.5, beta=2.0, epsilon=1.0, n_points=100)
    >>> plt.plot(x, y)
    >>> plt.axis('equal')
    >>> plt.xlabel('Position')
    >>> plt.ylabel('Momentum')
    >>> plt.show()

    See Also
    --------
    plot_ellipse : Higher-level function for plotting ellipses
    fit_ellipse : Fit ellipse parameters from data
    """
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
) -> matplotlib.axes.Axes:
    """Plot an ellipse in phase space.

    Creates a visualization of the ellipse defined by the given
    Courant-Snyder parameters. The ellipse is plotted with a label showing
    the parameter values.

    Parameters
    ----------
    alpha
        Courant-Snyder alpha parameter (dimensionless).
    beta
        Courant-Snyder beta parameter. Units determine x-axis scale.
    epsilon
        Emittance parameter. Determines the size of the ellipse.
    n_points
        Number of points to use when drawing the ellipse. More points create
        a smoother curve. Default is 400.
    ax
        Matplotlib axes on which to draw. If None, uses the current axes
        (via ``plt.gca()``).
    **kwargs_plot
        Additional keyword arguments passed to ``ax.plot()``, such as
        ``color``, ``linewidth``, ``linestyle``, etc.

    Returns
    -------
    artists
        The result of plt.plot(...).

    Examples
    --------
    Plot a simple ellipse:

    >>> import matplotlib.pyplot as plt
    >>> from blond.acc_math.analytic.ellipse import plot_ellipse
    >>>
    >>> fig, ax = plt.subplots()
    >>> plot_ellipse(alpha=0.5, beta=2.0, epsilon=1.0, ax=ax, color='blue')
    >>> ax.set_xlabel('Position [m]')
    >>> ax.set_ylabel('Momentum [rad]')
    >>> ax.legend()
    >>> plt.show()

    Plot multiple ellipses on the same axes:

    >>> fig, ax = plt.subplots()
    >>> for eps in [0.5, 1.0, 1.5]:
    ...     plot_ellipse(alpha=0.5, beta=2.0, epsilon=eps, ax=ax)
    >>> ax.set_xlabel('Position')
    >>> ax.set_ylabel('Momentum')
    >>> ax.legend()
    >>> plt.show()

    See Also
    --------
    get_points_on_ellipse : Generate ellipse coordinates without plotting
    fit_ellipse : Fit ellipse parameters from data
    """
    x, y = get_points_on_ellipse(alpha, beta, epsilon, n_points)

    if ax is None:
        ax = plt.gca()

    return ax.plot(
        x, y, label=f"α={alpha}, β={beta}, ε={epsilon}", **kwargs_plot
    )


def transform_twiss(
    x: NumpyArray,
    y: NumpyArray,
    alpha1: float,
    beta1: float,
    eps1: float,
    alpha2: float,
    beta2: float,
    eps2: float,
) -> tuple[NumpyArray, NumpyArray]:
    """Transform coordinates between two different Twiss parameter sets.

    This function transforms particle coordinates from one ellipse
    (characterized by alpha1, beta1, eps1) to another (characterized by alpha2,
    beta2, eps2). This is useful for matching beam distributions between different
    lattice sections or for beam preparation.

    Parameters
    ----------
    x
        Array of x-coordinates (e.g., position or time in phase space).
    y
        Array of y-coordinates (e.g., momentum or energy in phase space).
        Must have the same length as x.
    alpha1
        Courant-Snyder alpha parameter of the initial distribution (dimensionless).
    beta1
        Courant-Snyder beta parameter of the initial distribution.
    eps1
        Emittance of the initial distribution.
    alpha2
        Courant-Snyder alpha parameter of the target distribution (dimensionless).
    beta2
        Courant-Snyder beta parameter of the target distribution.
    eps2
        Emittance of the target distribution.

    Returns
    -------
    x2
        Transformed position coordinates in the new distribution.
    y2
        Transformed momentum coordinates in the new distribution.

    Notes
    -----
    The transformation is performed using the transfer matrix M = A2 · A1⁻¹, where:

    - A1 and A2 are the ellipse transformation matrices for the initial and target
      distributions, respectively.
    - The transformation matrix M maps coordinates from the initial to the target
      ellipse.

    This preserves the particle distribution shape but rescales and rotates it to
    match the new Twiss parameters and emittance.

    Examples
    --------
    Transform a beam distribution to match a different lattice section:

    >>> import numpy as np
    >>> from blond.acc_math.analytic.ellipse import transform_twiss
    >>>
    >>> # Initial beam coordinates
    >>> x = np.random.randn(10000) * 1e-3
    >>> xp = np.random.randn(10000) * 1e-6
    >>>
    >>> # Transform from initial to target Twiss parameters
    >>> x_new, xp_new = transform_twiss(
    ...     x, y,
    ...     alpha1=0.5, beta1=2.0, eps1=1e-6,  # initial
    ...     alpha2=-0.3, beta2=5.0, eps2=1e-6,  # target
    ... )

    See Also
    --------
    fit_ellipse : Determine Twiss parameters from particle coordinates
    twiss_from_cloud : Compute Twiss parameters from a particle distribution
    """
    A1 = np.sqrt(eps1) * np.array(
        [[np.sqrt(beta1), 0], [-alpha1 / np.sqrt(beta1), 1 / np.sqrt(beta1)]]
    )

    A2 = np.sqrt(eps2) * np.array(
        [[np.sqrt(beta2), 0], [-alpha2 / np.sqrt(beta2), 1 / np.sqrt(beta2)]]
    )

    M = A2 @ np.linalg.inv(A1)

    x2, y2 = M @ np.vstack([x, y])

    return x2, y2


def twiss_from_cloud(
    x: NumpyArray,
    xp: NumpyArray,
) -> tuple[float, float, float, float]:
    """Compute Twiss parameters from a particle distribution (point cloud).

    This function calculates the Courant-Snyder (Twiss) parameters and RMS emittance
    from a set of particle coordinates in phase space. The parameters are derived
    from the second-order moments of the distribution.

    Parameters
    ----------
    x
        Position coordinates (or time coordinates) of particles. Can be any shape,
        but will be flattened for moment calculation.
    xp
        Momentum coordinates (or energy coordinates) of particles. Must have the
        same shape as x.

    Returns
    -------
    beta : float
        Courant-Snyder beta parameter, computed from RMS position spread and emittance.
    alpha : float
        Courant-Snyder alpha parameter (dimensionless), computed from the correlation
        between position and momentum.
    gamma : float
        Courant-Snyder gamma parameter, computed from RMS momentum spread and emittance.
        Satisfies the relation ``gamma = (1 + alpha²) / beta``.
    eps : float
        RMS emittance of the distribution, computed as ``√(<x²><xp²> - <x·xp>²)``.

    Notes
    -----
    The Twiss parameters are calculated using the standard formulas:

    - ``ε = √(<x²><xp²> - <x·xp>²)`` (RMS emittance)
    - ``β = <x²> / ε``
    - ``α = -<x·xp> / ε``
    - ``γ = <xp²> / ε``

    where ``<·>`` denotes the mean over all particles.

    These formulas assume the distribution is centered (mean position and momentum
    are zero). For off-center distributions, subtract the means before calling this
    function.

    Examples
    --------
    Compute Twiss parameters from a Gaussian beam distribution:

    >>> import numpy as np
    >>> from blond.acc_math.analytic.ellipse import twiss_from_cloud
    >>>
    >>> # Generate a Gaussian distribution
    >>> n_particles = 100000
    >>> x = np.random.randn(n_particles) * 1e-3
    >>> xp = np.random.randn(n_particles) * 1e-6
    >>>
    >>> # Compute Twiss parameters
    >>> beta, alpha, gamma, eps = twiss_from_cloud(x, xp)
    >>> print(f"β = {beta:.3e}, α = {alpha:.3f}, γ = {gamma:.3e}, ε = {eps:.3e}")

    Compute parameters from a matched beam distribution:

    >>> from blond.acc_math.analytic.ellipse import get_points_on_ellipse
    >>>
    >>> # Create matched distribution with known parameters
    >>> x, xp = get_points_on_ellipse(alpha=0.5, beta=2.0, epsilon=1e-6, n_points=1000)
    >>> beta_fit, alpha_fit, gamma_fit, eps_fit = twiss_from_cloud(x, xp)
    >>> # Should recover the input parameters (approximately)

    See Also
    --------
    fit_ellipse : Alternative method using least-squares fitting
    transform_twiss : Transform coordinates between different Twiss parameter sets
    calc_ellipse_gamma : Relation between alpha, beta, and gamma
    """
    # compute second moments
    x2 = np.mean(x**2)
    y2 = np.mean(xp**2)
    xy = np.mean(x * xp)

    # rms emittance
    eps = np.sqrt(x2 * y2 - xy**2)

    # Twiss parameters
    beta = x2 / eps
    alpha = -xy / eps
    gamma = y2 / eps

    return beta, alpha, gamma, eps
