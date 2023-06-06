# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
:Authors: **Danilo Quartullo**, **Joel Repond**
'''

from __future__ import division, print_function

import numpy as np
import scipy.special as scisp
from scipy.constants import e


def analytical_gaussian_resonator(sigma_t, Q, R_s, omega_r, tau_array, n_particles):
    r"""Calculate the analytical induced voltage for a gaussian bunch and a resonator.

    Parameters
    ----------
    sigma_t : float
        RMS of the beam longitudinal coordinates [s].
    Q : float
        Quality factor of the resonator [1].
    R_s : float
        Shunt impedance of the resonator [:math:`\Omega`].
    omega_r : float
        Angular resonant frequency [rad/s].
    tau_array : float array
        Time array [s] for which the induced voltage should be calculated.
    n_particles : int
        Beam intensity [1].

    Returns
    -------
    induced_voltage : float array
        Induced voltage [V] (multiplied by -1 according to BLonD conventions).

    Notes
    -----
    The formula can be derived using the following lines of codes in
    Mathematica:

    .. code-block:: mathematica

        lambda = Exp[-(tau - t)^2 / (2 * sigma^2)] / ((2*Pi)^0.5 * sigma)
        wake = 2 * alpha * Rs * Exp[-alpha * t] * (Cos[ombar * t] - alpha / ombar
                                                   * Sin[ombar * t])
        output = Integrate[wake * lambda, {t, 0, Infinity},
                           Assumptions -> {sigma > 0, alpha > 0, Rs > 0, ombar > 0, tau in Reals}]
        outputreal = Simplify[ComplexExpand[Re[output]],
                              {sigma > 0, alpha > 0, Rs > 0, ombar > 0, tau in  Reals}]


    The imaginary part of output is identically equal to zero even if
    Mathematica cannot see that. To verify that the expression is correct
    launch:

    .. code-block:: mathematica

        output2 = Integrate[wake * lambda, t]
        outputreal2 = Simplify[ComplexExpand[Re[output2]],
                               {sigma > 0, alpha > 0, Rs > 0, ombar > 0, tau in Reals, t > 0}]
        Simplify[ComplexExpand[Re[d/dt output2]],
                 {sigma > 0, alpha > 0, Rs > 0, ombar > 0, tau in Reals, t > 0}]


    and check that applying the fundamental calculus theorem to outputreal2,
    one obtains back outputreal; the formula [5] in

    H. E. Salzer, "Formulas for Calculating the Error Function of a Complex
    Variable", Mathematical Tables and Other Aids to Computation, Vol. 5,
    No. 34 (Apr., 1951),pp. 67-70

    can be useful.

    """
    alpha = omega_r / (2 * Q)
    ombar = np.sqrt(omega_r**2 - alpha**2)

    A = (alpha * sigma_t**2 - tau_array + 1j * ombar * sigma_t**2) / (np.sqrt(2) * sigma_t)
    B = alpha * ombar * sigma_t**2 - ombar * tau_array
    result = R_s * alpha / ombar * np.e**(0.5 * (alpha**2 - ombar**2) * sigma_t**2 - alpha * tau_array) *\
        (scisp.erfc(A).real * (ombar * np.cos(B) + alpha * np.sin(B)) +
         scisp.erfc(A).imag * (alpha * np.cos(B) - ombar * np.sin(B)))

    induced_voltage = -n_particles * e * result

    return induced_voltage
