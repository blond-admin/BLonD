"""
Small shared utilities for the muon-collider feedback tests.

Numeric helpers (:func:`rel_err`, :func:`lab_frame_voltage`) and the opt-in
debug-plot boilerplate (:func:`open_debug_plot` / :func:`save_debug_plot`)
shared across the test modules in this package.
"""

import os

import numpy as np

# Debug-plot toggle for the test suite. Off by default
DEBUG_PLOTS: str | None = None


def rel_err(a, b) -> float:
    """
    Relative L2 error ``||a - b|| / ||b||``.

    Parameters
    ----------
    a
        Test array.
    b
        Reference array used to normalize the error.

    Returns
    -------
    float
        Relative L2 error of ``a`` against ``b``.
    """
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def lab_frame_voltage(v_ant, omega_rf, time, *, use_real: bool = False):
    """
    Project the complex antenna-voltage envelope back to the real lab frame.

    By default this is the ``external_reference=True`` / ``+pi/2`` demodulation
    convention used by
    :func:`blond.physics.feedbacks.helpers.rf_beam_current`; with
    ``use_real=True`` it is the ``external_reference=False`` convention.

    Parameters
    ----------
    v_ant
        Complex antenna-voltage envelope.
    omega_rf
        RF angular frequency used for the demodulation rotation.
    time
        Time base at which to evaluate the projection.
    use_real
        If True, return ``-Re[...]``; otherwise return ``-Im[...]``.

    Returns
    -------
    numpy.ndarray
        Real lab-frame voltage.
    """
    rotated = v_ant * np.exp(1j * omega_rf * time)
    return -(np.real(rotated) if use_real else np.imag(rotated))


def open_debug_plot():
    """
    Return ``(pyplot, mode)`` for an opt-in debug figure, or ``(None, None)``.

    Plotting is controlled by the module-level :data:`DEBUG_PLOTS` constant
    (set near the top of this module): ``"save"`` writes a PNG, ``"show"``
    also opens an interactive window, and ``None`` (the default) disables
    plotting. When enabled, the matplotlib backend is forced to a headless
    ``Agg`` unless ``mode == "show"``.

    Returns
    -------
    pyplot
        The ``matplotlib.pyplot`` module, or ``None`` when plotting is disabled.
    mode
        The :data:`DEBUG_PLOTS` value, or ``None`` when plotting is disabled.
    """
    mode = DEBUG_PLOTS
    if not mode:
        return None, None
    import matplotlib

    if mode != "show":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, mode


def save_debug_plot(fig, directory: str, filename: str, mode: str) -> None:
    """
    Save ``fig`` to ``directory/filename``; show if ``mode == "show"``; close.

    Parameters
    ----------
    fig
        Matplotlib figure to save.
    directory
        Output directory; typically ``os.path.dirname(__file__)`` of the
        calling test module, so the PNG lands next to the test.
    filename
        Output file name for the saved figure.
    mode
        If ``"show"``, also display the figure interactively before closing.
    """
    import matplotlib.pyplot as plt

    out = os.path.join(directory, filename)
    fig.savefig(out, dpi=120)
    print(f"\n[debug plot] saved to {out}")
    if mode == "show":
        plt.show()
    plt.close(fig)
