# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `PythonSpecials` and helper functions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import Specials
from blond.core.beam.flags import BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


# The function definition is recycled by the numba backend.
def _move_flagged_elements_to_end_py(
    flag: int,
    flags: NumpyArray,  # also purged
    dt: NumpyArray,
    dE: NumpyArray,
    ids: NumpyArray,
):
    """
    Reorder entries where ``flags == flag`` to the array end.

    This is only intended for `purge_flagged_entries`.

    Parameters
    ----------
    flag
        The flag to be used as a selector what to place at the end.
    flags
        Macro-particle flags.
    dt
        Macro-particle time coordinates [s].
    dE
        Macro-particle energy coordinates [eV].
    ids
        Macro-particle ids.
        This allows to identify single particles,
        even if the array indexing is changed.

    Returns
    -------
    n_new
        Number of particles that are not flagged.
    """
    i = 0
    j = flags.size - 1

    while i <= j:
        if flags[i] != flag:
            i += 1
        else:
            # flags[i] is True, swap with flags[j]
            flags[i], flags[j] = flags[j], flags[i]
            dt[i], dt[j] = dt[j], dt[i]
            dE[i], dE[j] = dE[j], dE[i]
            ids[i], ids[j] = ids[j], ids[i]
            j -= 1
    return j + 1


class PythonSpecials(Specials):
    """Implementation of backend functions in Python."""

    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        """
        Calculate the beam phase.

        Parameters
        ----------
        hist_x
            X axis of the histogram, usually in [s].
        hist_y
            Y axis of the histogram.
        alpha
            # TODO ported from blond2, was undocumented.
        omega_rf
            # TODO ported from blond2, was undocumented.
        phi_rf
            # TODO ported from blond2, was undocumented.
        bin_size
            # TODO ported from blond2, was undocumented.

        Returns
        -------
        beam_phase
            # TODO ported from blond2, was undocumented.
        """
        scoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.sin(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )
        ccoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.cos(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )

        return scoeff / ccoeff

    @staticmethod
    def histogram(
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: float,
        stop: float,
    ) -> None:
        """
        Calculate the histogram of an array.

        Parameters
        ----------
        array_read
            Array of many entries that should be compressed to a histogram.
        array_write
            Memory of where to write the histogram.
        start
            Start of the histogram bins.
        stop
            Stop of the histogram bins.
        """

        array_write[:], _ = np.histogram(
            array_read,
            range=(float(start), float(stop)),
            bins=len(array_write),
        )

    @staticmethod
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: NumpyArray,
        dE: NumpyArray,
        flags: NumpyArray,
    ) -> None:
        # select particles outside box
        select = (dE > e_max) | (dE < e_min) | (dt < t_min) | (dt > t_max)
        flags[select] = BeamFlags.LOST.value

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        """
        Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            RF voltage of the RF station, in [V].
        omega_rf
            Angular frequency of the RF system, in [rad/s].
        phi_rf
            RF station's design phase (per harmonic) in [rad].
        charge
            Particle charge, as number of elementary charges `e` [].
        acceleration_kick
            Energy that is added to all particles, in [eV].
        """

        voltage_kick = charge * voltage

        dE[:] += (
            voltage_kick * np.sin(omega_rf * dt[:] + phi_rf)
            + acceleration_kick
        )

    @staticmethod
    def kick_multi_harmonic(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        """
        Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            RF voltages of the RF station, in [V].
        omega_rf
            Angular frequencies of the RF system, in [rad/s].
        phi_rf
            RF station's design phases (per harmonic) in [rad].
        charge
            Particle charge, as number of elementary charges `e` [].
        n_rf
            Number of RF systems.
        acceleration_kick
            Energy that is added to all particles, in [eV].
        """
        voltage_kick = charge * voltage

        for j in range(n_rf):
            dE += voltage_kick[j] * np.sin(omega_rf[j] * dt + phi_rf[j])

        dE[:] += acceleration_kick

    @staticmethod
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        r"""
        Function to apply drift equation of motion.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        T
            Time spend in the drift region, in [s].
            :math:`T = L / (\beta c_0)`.
        eta_0
            General synchrotron parameter (zeroth-order slippage factor) [unitless].
        beta
            Relativistic velocity factor :math:`\beta = v/c` [unitless].
        energy
            Total beam energy [eV].
        """
        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = eta_0 / (beta * beta * energy)
        dt += T * coeff * dE

    @staticmethod
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        higher_alpha: NumpyArray,
        beta: float,
        energy: float,
    ) -> None:  # pragma: no cover
        r"""
        Exact drift equation of motion with higher order momentum compaction factors.

        Parameters
        ----------
        dt : NumpyArray
            Macro-particle time coordinates, in [s].
        dE : NumpyArray
            Macro-particle energy coordinates, in [eV].
        T : float
            Revolution period, in [s].
        alpha_0 : float
            Momentum compaction factor [unitless].
        higher_alpha : NumpyArray
            Momentum compaction factor to higher orders.
        beta
            Relativistic velocity factor :math:\beta = v/c [unitless].
        energy
            Total beam energy [eV].
        """
        n_alpha = len(higher_alpha)
        invbetasq = 1.0 / (beta * beta)
        inv_energy = 1.0 / energy
        inv_energy_sq = inv_energy * inv_energy

        # delta (vectorized)
        beam_delta = (
            np.sqrt(
                1.0
                + invbetasq * (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)
            )
            - 1.0
        )

        # ---- Polynomial evaluation ----
        poly = 1.0 + alpha_0 * beam_delta

        if n_alpha > 0:
            delta_power = beam_delta * beam_delta  # δ²

            for k in range(n_alpha):
                poly += higher_alpha[k] * delta_power
                delta_power *= beam_delta  # next power

        # ---- Final update ----
        dt += T * (poly * (1.0 + dE * inv_energy) / (1.0 + beam_delta) - 1.0)

    @staticmethod
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        """
        Interpolated kick method.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            Array of voltages along `bin_centers`, in [V].
        bin_centers
            Positions of `voltage`, in [s].
        charge
            Particle charge, as number of elementary charges `e` [].
        acceleration_kick
            Energy, in [eV], which is added to all particles.
            This is intended to subtract the target energy from the RF
            energy gain in one common call.
        """
        n_slices = len(bin_centers)
        inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

        fbin = np.floor((dt - bin_centers[0]) * inv_bin_width).astype(np.int32)

        helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
        helper2 = (
            charge * voltage[:-1] - bin_centers[:-1] * helper1
        ) + acceleration_kick

        for i in range(len(dt)):
            # fbin = int(np.floor((dt[i]-bin_centers[0])*inv_bin_width))
            if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
                dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]

    @staticmethod
    def move_flagged_elements_to_end(
        flag: int,
        flags: NumpyArray | CupyArray,  # also purged
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        ids: NumpyArray | CupyArray,
    ):
        """
        Reorder entries where ``flags == flag`` to the array end.

        This is only intended for `purge_flagged_entries`.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
        flags
            Macro-particle flags.
        dt
            Macro-particle time coordinates [s].
        dE
            Macro-particle energy coordinates [eV].
        ids
            Macro-particle ids.
            This allows to identify single particles,
            even if the array indexing is changed.

        Returns
        -------
        n_new
            Number of particles that are not flagged.
        """
        n_new = _move_flagged_elements_to_end_py(
            flag=np.int32(flag),
            flags=flags,
            dt=dt,
            dE=dE,
            ids=ids,
        )
        return n_new

    @staticmethod
    def fused_kick_drift_profile(
        dt,
        dE,
        voltage,
        phi_rf,
        omega_rf,
        charge,
        acceleration_kick,
        T,
        eta_0,
        beta,
        energy,
    ):
        voltage_kick = charge * voltage

        dE[:] += (
            voltage_kick * np.sin(omega_rf * dt[:] + phi_rf)
            + acceleration_kick
        )
        coeff = eta_0 / (beta * beta * energy)
        dt[:] += T * coeff * dE