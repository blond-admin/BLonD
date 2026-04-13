# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Implementation of barrier bucket RF system."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import scipy.constants as cont

from blond.core.backends.backend import backend
from blond.generals.exceptions_ import ArrayShapeError
from blond.physics.cavities import RFManipulationBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass


class BarrierRF(RFManipulationBaseClass):
    """
    Define waveforms for a barrier bucket RF system.

    Class to generate barrier bucket RF systems.  Converts parameters
    for a given barrier to suitable inputs for an RFStation object.

    Based on developments of M. Vadai for PS barrier bucket system [1].

    Parameters
    ----------
    t_center
        The center time of the barrier, in [s].
    t_width
        The width the barrier, in [s].
    peak_voltage
        The peak amplitude of the barrier, in [V].
    n_bins
        If tracking directly, specifies the number of bins that will
        define the waveform before interpolation.
    section_index
        The section the barrier should be applied to.

    References
    ----------
    [1] M. Vadai, et al, "Beam Manipulations With Barrier Buckets in the
    CERN PS", https://cds.cern.ch/record/2694233/files/mopts107.pdf
    """

    def __init__(
        self,
        t_center: float | None = None,
        t_width: float | None = None,
        peak_voltage: float | None = None,
        n_bins: int | None = None,
        section_index: int = 0,
    ):
        super().__init__(section_index=section_index)

        self.t_center: float = t_center
        self.t_width: float = t_width
        self.peak: float = peak_voltage
        self.n_bins = n_bins

        self._add_intended_schedule("t_center", "t_width", "peak_voltage")

    def waveform_at_turn_or_time(
        self, turn_i: int, reference_time: float, bin_centers: ArrayLike
    ) -> NumpyArray | CupyArray:
        """
        Create the barrier waveform at the specified time.

        Construct the ideal barrier waveform at the specified time on
        the given bin_centers

        Parameters
        ----------
        turn_i
            The turn number at which to compute_the waveform.
        reference_time
            The time in the ramp at which to compute the waveform.
        bin_centers
            The bin centers for the timespan to cover with the waveform.

        Returns
        -------
        waveform
            The array of the barrier waveform.
        """
        self.apply_schedules(turn_i, reference_time)

        return compute_sin_barrier(
            self.t_center, self.t_width, self.peak, bin_centers
        )

    def to_fourier_series(
        self,
        t_rev: Iterable[float],
        harmonics: Iterable[int],
        turns: Iterable[int] | None = None,
        times: Iterable[float] | None = None,
        filter_order: int = 1,
    ) -> tuple[
        list[int], list[NumpyArray | CupyArray], list[NumpyArray | CupyArray]
    ]:
        """
        Convert the barrier definition into a Fourier series.

        The barrier waveform will be constructed at all given times and
        converted to a Fourier series of amplitude and phase for each
        harmonic at those times.

        Parameters
        ----------
        t_rev
            The revolution time at the times of interest, in [s].
        harmonics
            The RF harmonics used for the Fourier series.
        turns
            The turns at which to construct the Fourier series.
        times
            The times at which to construct the Fourier series.
        filter_order
            The order of the sinc filter to be applied.  For details,
            see sinc_filtering function.  Defaults to 1.

        Returns
        -------
        harmonics, voltages, phases
            A tuple containing:
                The original input harmonics as a list.
                A list of arrays defining the voltage at the requested turns/times, in [V].
                A list of arrays defining the phase at the requested turns/times, in [rad].

        Raises
        ------
            ValueError: Raised if len(times) != len(t_rev)
        """
        match (turns, times):
            case None, None:
                raise ValueError(
                    "At least one of turns or times must be supplied"
                )
            case Iterable(), None:
                turns = list(turns)
                times = [None for _ in turns]
            case None, Iterable():
                times = list(times)
                turns = [None for _ in times]
            case Iterable(), Iterable():
                times = list(times)
                turns = list(turns)
                if len(times) != len(turns):
                    raise ValueError(
                        "If specifying both turns and times, the same number "
                        "of elements must be given for both."
                    )

        max_h = backend.max(harmonics)

        # Should not be possible to enter, kept for safety
        if len(times) != len(t_rev):  # pragma: no cover
            raise ValueError(
                "Input times and t_rev must have the same"
                + " number of elements"
            )

        voltages = []
        phases = []
        harmonics = list(harmonics)

        for _ in harmonics:
            v = backend.zeros(len(times))
            p = backend.zeros(len(times))
            voltages.append(v)
            phases.append(p)

        for i, (tn, tm, tr) in enumerate(
            zip(turns, times, t_rev, strict=False)
        ):
            # Used 10*max_h to go well above the Nyquist frequency,
            # exact value not important
            bin_width = tr / (10 * max_h)
            n_bins = int(tr / bin_width)
            bin_cents = backend.linspace(0, tr, n_bins)
            barrier = self.waveform_at_turn_or_time(tn, tm, bin_cents)

            amps, phis = waveform_to_harmonics(barrier, harmonics)
            amps = sinc_filtering(amps, filter_order)

            g_comp = _gain_compensation(
                bin_cents, barrier, harmonics, amps, phis
            )

            amps /= g_comp

            for j in range(len(harmonics)):
                voltages[j][i] = amps[j]
                phases[j][i] = phis[j]

        return harmonics, voltages, phases

    def _track(self, beam: BeamBaseClass) -> None:
        """
        Main simulation routine to be called in the mainloop.

        Parameters
        ----------
        beam
            Beam class to interact with this element.
        """
        super()._track(beam=beam)

        turn = self._turn_i.value
        time = beam.reference.time

        beta = beam.reference.beta
        circ = self._ring.circumference

        t_rev = circ / (beta * cont.c)
        bins = backend.linspace(0, t_rev, self.n_bins, dtype=backend.float)

        waveform = self.waveform_at_turn_or_time(turn, time, bins)

        reference = beam.reference
        reference_energy_change = self.track_reference(
            reference, beam.is_counter_rotating
        )

        # TODO: Integrate with `PooledInterpolationKick`
        backend.specials.kick_induced_voltage(
            beam.write_partial_dt(),
            beam.write_partial_dE(),
            waveform,
            bins,
            beam.particle_type.charge,
            acceleration_kick=-reference_energy_change,
        )

    def on_run_simulation(self, simulation, beam, n_turns, **kwargs):
        """
        Lateinit method when `simulation.run_simulation` is called.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation `Beam` object.
        n_turns
            Number of turns to simulate.
        **kwargs
            Additional keyword arguments.
        """
        super().on_run_simulation(simulation, beam, n_turns, **kwargs)


def compute_sin_barrier(
    center: float,
    width: float,
    amplitude: float,
    bin_centers: Iterable[float],
    periodic: bool = True,
) -> NumpyArray | CupyArray:
    """
    Compute a single-period sinusoidal barrier.

    Parameters
    ----------
    center
        The time-center of the barrier, in [S].
    width
        The width of the barrier, in [S].
    amplitude
        The peak amplitude of the barrier, in [V].
    bin_centers
        The bin centers to use, in [s].
    periodic
        If the barrier voltage extends below `bin_centers[-1]` or above
        `bin_centers[0]`, the periodic flag wraps it around to the
        other end of the window.
        Defaults to True.

    Returns
    -------
    barrier_waveform
        An array of the barrier waveform.

    Raises
    ------
    ValueError
        Raises a ValueError if the barrier is longer than the given
        bin_centers.
    """
    bin_centers = backend._asarray_if_needed(bin_centers)
    if len(bin_centers.shape) != 1:
        raise ArrayShapeError("bin_centers array must be 1-dimensional.")

    barrier_waveform = backend.zeros_like(bin_centers, dtype=backend.float)

    t_step = bin_centers[1] - bin_centers[0]
    n_bins = int(width / t_step)
    barr_time = backend.linspace(
        center - width / 2, center + width / 2, n_bins
    )

    if barr_time[-1] - barr_time[0] > bin_centers[-1] - bin_centers[0]:
        raise ValueError("Given barrier width is too large and will overflow")

    barrier = amplitude * backend.sin(
        2 * backend.pi * (barr_time - center) / width
    )

    barrier_waveform += backend.interp(
        bin_centers, barr_time, barrier, left=0, right=0
    )
    if periodic:
        if barr_time[-1] > bin_centers[-1]:
            barrier_waveform += backend.interp(
                bin_centers,
                barr_time - bin_centers[-1],
                barrier,
                left=0,
                right=0,
            )
        if barr_time[0] < bin_centers[0]:
            barrier_waveform += backend.interp(
                bin_centers,
                barr_time + bin_centers[-1],
                barrier,
                left=0,
                right=0,
            )

    return barrier_waveform


def harmonics_to_waveform(
    bin_centers: Iterable[float],
    harmonic_numbers: Iterable[int],
    harmonic_amplitudes: Iterable[float],
    harmonic_phases: Iterable[float],
    t_rev: float | None = None,
) -> NumpyArray | CupyArray:
    """
    Convert a Fourier series to a waveform.

    Parameters
    ----------
    bin_centers
        The bin centers at which to compute the waveform.
    harmonic_numbers
        The harmonic numbers of the Fourier series.
    harmonic_amplitudes
        The amplitude of each harmonic of the Fourier series.
    harmonic_phases
        The phase of each harmonic of the Fourier series.
    t_rev
        The revolution time, defaults to None.  If None, it is assumed
        that the bin_centers cover a full turn.

    Returns
    -------
    waveform
        The reconstructed barrier waveform.
    """
    bin_centers = backend._asarray_if_needed(bin_centers)

    if t_rev is None:
        t_rev = bin_centers[-1] - bin_centers[0]

    waveform = backend.zeros_like(bin_centers, dtype=backend.float)
    for harm, amp, phi in zip(
        harmonic_numbers, harmonic_amplitudes, harmonic_phases, strict=False
    ):
        waveform += amp * backend.sin(
            harm * 2 * backend.pi * bin_centers / t_rev + phi
        )

    return waveform


def waveform_to_harmonics(
    waveform: ArrayLike, harmonics: Iterable[int] | None = None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Convert a waveform to a Fourier series.

    Converts an arbitrary waveform to a Fourier series in amplitude and
    phase. The waveform is assumed to be one revolution period in
    length. The harmonic numbers must be integers and are used to
    select the required Fourier components.

    The input waveform can be reconstructed with a `sin` function.

    Parameters
    ----------
    waveform
        Voltage waveform covering a single revolution period.
    harmonics
        The RF harmonics to be used for the final Fourier series.
        If None, all harmonics are used.

    Returns
    -------
    harmonic_amps, harmonic_phases
        Two tuples of float, length equal to len(harmonics).
        Element 0 is the amplitudes, element 1 is the phases.
    """
    harm_series = backend.fft.rfft(backend._asarray_if_needed(waveform))

    if harmonics is not None:
        harm_series = backend.array(
            [harm_series[h] for h in backend._asarray_if_needed(harmonics)],
            dtype=backend.complex,
        )

    harm_amps = backend.abs(harm_series) / (len(waveform) / 2)
    harm_phases = (
        backend.arctan2(harm_series.real, harm_series.imag) + backend.pi
    )

    return harm_amps, harm_phases


def sinc_filtering(
    harmonic_amplitudes: Iterable[float], filter_order: int = 1
) -> NumpyArray:
    """
    Sinc filtering of a Fourier series.

    Filters the Fourier components with a sinc function window as
    described in [1].

    Parameters
    ----------
    harmonic_amplitudes
        The amplitudes of the Fourier series.  Assumed to be
        uniformly spaced in the range 1..n.
    filter_order
        Power applied to the sinc function. Higher values give more
        aggressive filtering, 0 is equivalent to a square window, or
        no filtering.
        Defaults to 1.

    Returns
    -------
    filtered_amplitudes
        The modified harmonic amplitudes.

    References
    ----------
    [1] M. Vadai, "Beam Loss Reduction by Barrier Buckets in the CERN
    Accelerator Complex", CERN-THESIS-2021-043 (Chapter 3.2.3.2).
    """
    filtered_amplitudes = backend.zeros_like(
        harmonic_amplitudes, dtype=backend.float
    )
    n_harm = len(harmonic_amplitudes)

    for i, a in enumerate(harmonic_amplitudes):
        filtered_amplitudes[i] = (
            a
            * backend.sinc(((i + 1) * backend.pi) / (2 * (n_harm + 1)))
            ** filter_order
        )

    return filtered_amplitudes


def _gain_compensation(
    barrier_time: NumpyArray,
    barrier_waveform: NumpyArray,
    harmonics: NumpyArray,
    harmonic_amplitudes: NumpyArray,
    harmonic_phases: NumpyArray,
    t_rev: float | None = None,
) -> NumpyArray:
    reconstructed = harmonics_to_waveform(
        barrier_time, harmonics, harmonic_amplitudes, harmonic_phases, t_rev
    )

    ratio_max = backend.max(reconstructed) / backend.max(barrier_waveform)
    ratio_min = backend.abs(
        backend.min(reconstructed) / backend.min(barrier_waveform)
    )

    return ratio_max if ratio_max > ratio_min else ratio_min
