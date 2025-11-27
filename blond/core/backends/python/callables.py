# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `PythonSpecials` and helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import Specials

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
    """Reorders entries where ``flags == flag`` to the array end.

    This is only intended for `purge_flagged_entries`.

    Parameters
    ----------
    flag
        The flag to be used as a selector what to place at the end.
    flags
        Macro-particle flags
    dt
        Macro-particle time coordinates [s]
    dE
        Macro-particle energy coordinates [eV]
    ids
        Macro-particle ids.
        This allows to identify single particles,
        even if the array indexing is changed.
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
    def meta_params_multibunch(
        dt: NumpyArray,
        dE: NumpyArray,
        mask: NumpyArray,
        sigma_dt_buffer: NumpyArray,
        sigma_dE_buffer: NumpyArray,
        mean_dt_buffer: NumpyArray,
        mean_dE_buffer: NumpyArray,
        rms_emittance_buffer: NumpyArray,
        t_rf: float,
    ) -> None:
        """
        Calculates mean and standard deviation of both energy and time coordinates as well as rms bunch emittance.

        Parameters
        ----------
        dt
            input array of time values [s]
        dE
            input array of energies [eV]
        mask
            mask to be used on dt and dE for the calculation of the meta parameters
        sigma_dt_buffer
            output buffer for standard deviation of time axis
        sigma_dE_buffer
            output buffer for standard deviation of energy axis
        mean_dt_buffer
            output buffer for mean of time axis
        mean_dE_buffer
            output buffer for mean of energy axis
        rms_emittance_buffer
            output buffer for rms emittance
        t_rf
            period of main RF, used to correct the mean of bunches, which are behind the first bucket

        Notes
        -----
        The mean of dt is corrected to the first bucket.

        """
        for bucket in range(len(mask)):
            sigma_dt_buffer[bucket] = np.std(dt[mask[bucket]])
            sigma_dE_buffer[bucket] = np.std(dE[mask[bucket]])
            mean_dt_buffer[bucket] = np.mean(dt[mask[bucket]]) - bucket * t_rf
            # correct to value of first bucket
            mean_dE_buffer[bucket] = np.mean(dE[mask[bucket]])
            dt_corrected_axis = dt[mask[bucket]] - bucket * t_rf
            rms_emittance_buffer[bucket] = np.sqrt(
                np.average(dE[mask[bucket]] ** 2)
                * np.average(dt_corrected_axis**2)
                - np.average(dE[mask[bucket]] * dt_corrected_axis) ** 2
            )

    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        """Calculates the beam phase.

        Parameters
        ----------
        hist_x
            x axis of the histogram, usually in [s].
        hist_y
            y axis of the histogram.
        alpha
            # TODO ported from blond2, was undocumented
        omega_rf
            # TODO ported from blond2, was undocumented
        phi_rf
            # TODO ported from blond2, was undocumented
        bin_size
            # TODO ported from blond2, was undocumented

        Returns
        -------
        beam_phase
            # TODO ported from blond2, was undocumented
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
        """Calculate the histogram of an array.

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
        top: float,
        bottom: float,
        left: float,
        right: float,
        dt: NumpyArray,
        dE: NumpyArray,
        flags: NumpyArray,
    ) -> None:
        # select particles outside box
        select = (dE > top) | (dE < bottom) | (dt < left) | (dt > right)
        flags[select] = -500  # assume (BeamFlags.LOST.value)

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
        """Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        voltage
            RF voltage of the RF station, in [V]
        omega_rf
            Angular frequency of the RF system, in [rad/s]
        phi_rf
            RF station's design phase (per harmonic) in [rad]
        charge
            Particle charge, as number of elementary charges `e` []
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
        """Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        voltage
            RF voltages of the RF station, in [V]
        omega_rf
            Angular frequencies of the RF system, in [rad/s]
        phi_rf
            RF station's design phases (per harmonic) in [rad]
        charge
            Particle charge, as number of elementary charges `e` []
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
        r"""Function to apply drift equation of motion.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        T
            Revolution period, in [s].
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
    def drift_legacy(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_order: int,
        eta_0: float,
        eta_1: float,
        eta_2: float,
        beta: float,
        energy: float,
    ) -> None:  # pragma: no cover # TODO
        r"""Function to apply drift equation of motion.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        T
            Revolution period, in [s].
        alpha_order
            Oder of the alpha parameter
        eta_0
            General synchrotron parameter (zeroth-order slippage factor) [unitless].
        eta_1
            General synchrotron parameter (zeroth-order slippage factor) [unitless].
        eta_2
            General synchrotron parameter (zeroth-order slippage factor) [unitless].
        beta
            Relativistic velocity factor :math:`\beta = v/c` [unitless].
        energy
            Total beam energy [eV].
        """
        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = 1.0 / (beta * beta * energy)
        eta0 = eta_0 * coeff
        eta1 = eta_1 * coeff * coeff
        eta2 = eta_2 * coeff * coeff * coeff

        if alpha_order == 0:
            dt += T * (1.0 / (1.0 - eta0 * dE) - 1.0)
        elif alpha_order == 1:
            dt += T * (1.0 / (1.0 - eta0 * dE - eta1 * dE * dE) - 1.0)
        else:
            dt += T * (
                1.0 / (1.0 - eta0 * dE - eta1 * dE * dE - eta2 * dE * dE * dE)
                - 1.0
            )

    @staticmethod
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        alpha_1: float,
        alpha_2: float,
        beta: float,
        energy: float,
    ) -> None:  # pragma: no cover # TODO
        r"""Function to apply drift equation of motion.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        T
            Revolution period, in [s].
        alpha_0
            Momentum compaction factor [unitless].
        alpha_1
            Momentum compaction factor [unitless].
        alpha_2
            Momentum compaction factor [unitless].
        beta
            Relativistic velocity factor :math:`\beta = v/c` [unitless].
        energy
            Total beam energy [eV].
        """
        # solver_decoded = solver.decode(encoding='utf_8')

        invbetasq = 1 / (beta * beta)
        invenesq = 1 / (energy * energy)
        # double beam_delta;

        beam_delta = (
            np.sqrt(1.0 + invbetasq * (dE * dE * invenesq + 2.0 * dE / energy))
            - 1.0
        )

        dt += T * (
            (
                1.0
                + alpha_0 * beam_delta
                + alpha_1 * (beam_delta * beam_delta)
                + alpha_2 * (beam_delta * beam_delta * beam_delta)
            )
            * (1.0 + dE / energy)
            / (1.0 + beam_delta)
            - 1.0
        )

    @staticmethod
    def kick_induced_voltage(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        """Interpolated kick method.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s]
        dE
            Macro-particle energy coordinates, in [eV]
        voltage
            Array of voltages along `bin_centers`, in [V]
        bin_centers
            Positions of `voltage`, in [s]
        charge
            Particle charge, as number of elementary charges `e` []
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
        """Reorders entries where ``flags == flag`` to the array end.

        This is only intended for `purge_flagged_entries`.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
        flags
            Macro-particle flags
        dt
            Macro-particle time coordinates [s]
        dE
            Macro-particle energy coordinates [eV]
        ids
            Macro-particle ids.
            This allows to identify single particles,
            even if the array indexing is changed.
        """
        n_new = _move_flagged_elements_to_end_py(
            flag=np.int32(flag),
            flags=flags,
            dt=dt,
            dE=dE,
            ids=ids,
        )
        return n_new
