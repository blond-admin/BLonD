"""
Small shared utilities for the muon-collider feedback tests.

Numeric helpers (:func:`rel_err`, :func:`lab_frame_voltage`)
shared across the test modules in this package.
"""

import numpy as np

from blond.generals.cupy.no_cupy_import import copy_to_cpu


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
    a = copy_to_cpu(np.asarray(a) if np.isscalar(a) else a)
    b = copy_to_cpu(np.asarray(b) if np.isscalar(b) else b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def lab_frame_voltage(v_ant, omega_rf, time, *, use_real: bool = False):
    """
    Project the complex antenna-voltage envelope back to the real lab frame.

    By default this is the ``external_reference=True`` / ``+pi/2`` demodulation
    convention used by
    :func:`blond.physics.feedbacks.beam_current.rf_beam_current`; with
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
    v_ant = copy_to_cpu(v_ant) if not np.isscalar(v_ant) else v_ant
    time = copy_to_cpu(time) if not np.isscalar(time) else time
    rotated = v_ant * np.exp(1j * omega_rf * time)
    return -(np.real(rotated) if use_real else np.imag(rotated))
