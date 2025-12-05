import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def ellipse_residuals(params, x, y):
    """
    Params = [xc, yc, a, b, theta]
    Computes residual of each point w.r.t ellipse equation
    ((x')/a)^2 + ((y')/b)^2 - 1 = 0
    where (x',y') are rotated into ellipse-aligned frame.
    """
    xc, yc, a, b, theta = params
    ct, st = np.cos(theta), np.sin(theta)

    # Shift
    xs = x - xc
    ys = y - yc

    # Rotate into ellipse axes
    x_rot = ct * xs + st * ys
    y_rot = -st * xs + ct * ys

    # Residuals of ellipse implicit equation
    return (x_rot / a) ** 2 + (y_rot / b) ** 2 - 1


def fit_ellipse_rotated(x, y):
    # initial guess
    xc0 = np.mean(x)
    yc0 = np.mean(y)
    a0 = (np.max(x) - np.min(x)) / 2
    b0 = (np.max(y) - np.min(y)) / 2
    theta0 = 0.0  # rotation guess

    p0 = [xc0, yc0, a0, b0, theta0]

    # Fit
    sol = least_squares(ellipse_residuals, p0, args=(x, y))

    xc, yc, a, b, theta = sol.x
    return float(xc), float(yc), abs(a), abs(b), float(theta)


def plot_fitted_ellipse(x, y):
    x = x / 1e-8
    y = y / 1e8
    xc, yc, a, b, theta = fit_ellipse_rotated(x, y)

    # Ellipse points
    t = np.linspace(0, 2 * np.pi, 400)
    ct, st = np.cos(theta), np.sin(theta)
    xe = xc + a * np.cos(t) * ct - b * np.sin(t) * st
    ye = yc + a * np.cos(t) * st + b * np.sin(t) * ct

    plt.plot(xe * 1e-8, ye * 1e8, "b-")


if __name__ == "__main__":
    # Example usage
    x = np.array([1, 2, -1.5])
    y = np.array([1, 2, 1.5])

    plot_fitted_ellipse(x, y)
