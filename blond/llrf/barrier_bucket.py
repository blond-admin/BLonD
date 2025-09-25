from __future__ import annotations

# General imports
import numpy as np
import abc
from typing import TYPE_CHECKING

try:
    import cupy as cp
except ImportError:
    _CUPY_AVAILABLE = False
else:
    _CUPY_AVAILABLE = True

# BLonD imports
from ..input_parameters import rf_parameters as rfpar
from ..utils import bmath as bm
from ..utils import data_check as dc


if TYPE_CHECKING:  # pragma: no cover
    from typing import Iterable, Optional

    from numpy.typing import NDArray as NumpyArray

    if _CUPY_AVAILABLE:
        from cupy.typing import NDArray as CupyArray

    from blond.beam.beam import Beam


class BarrierGenerator:
    """
    Class to generate barrier bucket RF systems.  Converts parameters
    for a given barrier to suitable inputs for an RFStation object.

    Based on developments of M. Vadai for PS barrier bucket system.
    https://cds.cern.ch/record/2694233/files/mopts107.pdf

    Parameters
    ----------
        t_center : (float | Iterable[Iterable[float]])
            The center time of the barrier in seconds, either constant
            or time varying.  If time varying, the format should be
            [time, amplitude].
        t_width : (float | Iterable[Iterable[float]])
            The width the barrier in seconds, either constant
            or time varying.  If time varying, the format should be
            [time, amplitude].
        peak : (float | Iterable[Iterable[float]])
            The peak amplitude of the barrier in V, either constant
            or time varying.  If time varying, the format should be
            [time, amplitude].
    """

    def __init__(
        self,
        t_center: float | Iterable[Iterable[float]],
        t_width: float | Iterable[Iterable[float]],
        peak: float | Iterable[Iterable[float]],
    ):
        self._input_t_center = t_center
        self._input_t_width = t_width
        self._input_peak = peak

    def waveform_at_time(
        self, time: float, bin_centers: Iterable[float]
    ) -> NumpyArray | CupyArray:
        """
        Construct the ideal barrier waveform at the specified time on
        the given bin_centers

        Args:
            time (float):
                The time in the ramp at which to compute the waveform.
            bin_centers (Iterable[float]):
                The bin centers for the timespan to cover with the
                waveform.

        Returns:
            CupyArray|NumpyArray: The array of the barrier waveform
        """
        # TODO:  Allow function type (sin, square, ...) as input

        cent = dc.interp_if_array(time, self._input_t_center)
        width = dc.interp_if_array(time, self._input_t_width)
        peak = dc.interp_if_array(time, self._input_peak)

        return compute_sin_barrier(cent, width, peak, bin_centers)

    def for_rf_station(
        self,
        times: Iterable[float],
        t_rev: Iterable[float],
        harmonics: Iterable[int],
        m: int = 1,
    ) -> tuple[list[int], list[NumpyArray], list[NumpyArray]]:
        """
        Converts the barrier definition into a form that can be input to
        the RFStation object.  The barrier will be constructed at all
        given times and converted to a Fourier series to give the
        amplitude and phase for each harmonic at those times.

        Args:
            times (Iterable[float]): The times at which to construct the
                                     Fourier series
            t_rev (Iterable[float]): The revolution time at the
            harmonics (Iterable[int]): The RF harmonics used for the
                                       barrier
            m (int, optional): The order of the sinc filter to be
                               applied.  For details, see sinc_filtering
                               function.
                               Defaults to 1.

        Raises:
            ValueError: Raised if len(times) != len(t_rev)

        Returns:
            tuple[list[int], list[NumpyArray], list[NumpyArray]]:
                    A tuple containing:
                        The original input harmonics as a list
                        A list of 2-arrays defining the voltages
                        A list of 2-arrays defining the phases
        """

        max_h = bm.max(harmonics)

        if len(times) != len(t_rev):
            raise ValueError(
                "Input times and t_rev must have the same"
                + " number of elements"
            )

        voltages = []
        phases = []
        harmonics = list(harmonics)

        for _ in harmonics:
            v = np.zeros([2, len(times)])
            p = np.zeros([2, len(times)])
            v[0] = times
            p[0] = times
            voltages.append(v)
            phases.append(p)

        for i, (time, tr) in enumerate(zip(times, t_rev)):
            bin_width = tr / (10 * max_h)
            n_bins = int(tr / bin_width)
            bin_cents = bm.linspace(0, tr, n_bins)
            barrier = self.waveform_at_time(time, bin_cents)

            amps, phis = waveform_to_harmonics(barrier, harmonics)
            amps = sinc_filtering(amps, m)

            g_comp = _gain_compensation(
                bin_cents, barrier, harmonics, amps, phis
            )

            amps /= g_comp

            for j in range(len(harmonics)):
                voltages[j][1, i] = amps[j]
                phases[j][1, i] = phis[j]

        return harmonics, voltages, phases


def compute_sin_barrier(
    center: float,
    width: float,
    amplitude: float,
    bin_centers: Iterable[float],
    periodic: bool = True,
) -> NumpyArray | CupyArray:
    """
    Computes a single-period sinusoidal barrier.

    Args:
        center (float): The time-center of the barrier
        width (float): The width of the barrier
        amplitude (float): The peak amplitude of the barrier
        bin_centers (Iterable[float]): The bin centers to use
        periodic (bool, optional):
                    Flag to enable barriers to wrap around at the
                    start/end of bin_centers.
                    Defaults to True.

    Raises:
        ValueError: Raises a ValueError if the barrier is longer than
                    the given bin_centers.

    Returns:
        NumpyArray | CupyArray: Either a cupy or numpy array of the
                                barrier waveform.
    """

    barrier_waveform = bm.zeros_like(bin_centers)

    t_step = bin_centers[1] - bin_centers[0]
    n_bins = int(width / t_step)
    barr_time = np.linspace(center - width / 2, center + width / 2, n_bins)

    if barr_time[-1] - barr_time[0] > bin_centers[-1] - bin_centers[0]:
        raise ValueError("Given barrier width is too large and will overflow")

    barrier = amplitude * bm.sin(2 * np.pi * (barr_time - center) / width)

    barrier_waveform += bm.interp(
        bin_centers, barr_time, barrier, left=0, right=0
    )
    if periodic:
        if barr_time[-1] > bin_centers[-1]:
            barrier_waveform += bm.interp(
                bin_centers,
                barr_time - bin_centers[-1],
                barrier,
                left=0,
                right=0,
            )
        if barr_time[0] < bin_centers[0]:
            barrier_waveform += bm.interp(
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
    t_rev: Optional[float] = None,
) -> NumpyArray:
    if t_rev is None:
        t_rev = bin_centers[-1] - bin_centers[0]

    waveform = np.zeros_like(bin_centers)
    for h, a, p in zip(harmonic_numbers, harmonic_amplitudes, harmonic_phases):
        waveform += a * np.sin(h * 2 * np.pi * bin_centers / t_rev + p)

    return waveform


def waveform_to_harmonics(
    waveform: NumpyArray | CupyArray, harmonics: Optional[Iterable[int]] = None
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """
    Converts an arbitrary waveform to a fourier series in amplitude and
    phase.  Waveform is assumed to be 1 revolution period in length.
    the harmonic numbers must be an integer and are used to select the
    required fourier components.

    The input waveform can be reconstructed with a sin function.

    Args:
        waveform (NumpyArray | CupyArray):
            Voltage waveform covering a single revolution period.
        harmonics (Optional[Iterable[int]]):
            The RF harmonics to be used for the final fourier series.
            If None, all harmonics are used.

    Returns:
        tuple[tuple[float, ...]]:
            Two tuples of float, length equal to len(harmonics).
            Element 0 is the amplitudes, element 1 is the phases.
    """

    harm_series = np.fft.rfft(waveform)

    if harmonics is not None:
        harm_series = np.array([harm_series[h] for h in harmonics])

    harm_amps = np.abs(harm_series) / (len(waveform) / 2)
    harm_phases = np.arctan2(harm_series.real, harm_series.imag) + np.pi

    return harm_amps, harm_phases


def sinc_filtering(
    harmonic_amplitudes: Iterable[float], m: int = 1
) -> NumpyArray:
    """
    Filters the fourier components with a sinc function window as
    described in PhD thesis:
        Beam Loss Reduction by Barrier Buckets in the CERN Accelerator
        Complex:  M. Vadai CERN-THESIS-2021-043 Chapter 3.2.3.2

    Args:
        harmonic_amplitudes (Iterable[float]):
            The amplitudes of the fourier series.  Assumed to be
            uniformly spaced in the range 1..n
        m (int, optional):
            Power applied to the sinc function.  Higher values give more
            aggressive filtering, 0 is equivalent to a square window, or
            no filtering.
            Defaults to 1.

    Returns:
        NDArray: The modified harmonic amplitudes.
    """

    filtered_amplitudes = np.zeros_like(harmonic_amplitudes)
    n_harm = len(harmonic_amplitudes)

    for i, a in enumerate(harmonic_amplitudes):
        filtered_amplitudes[i] = (
            a * np.sinc(((i + 1) * np.pi) / (2 * (n_harm + 1))) ** m
        )

    return filtered_amplitudes


def _gain_compensation(
    barrier_time: NumpyArray,
    barrier_waveform: NumpyArray,
    harmonics: NumpyArray,
    harmonic_amplitudes: NumpyArray,
    harmonic_phases: NumpyArray,
    t_rev: Optional[float] = None,
) -> NumpyArray:
    reconstructed = harmonics_to_waveform(
        barrier_time, harmonics, harmonic_amplitudes, harmonic_phases, t_rev
    )

    ratio_max = np.max(reconstructed) / np.max(barrier_waveform)
    ratio_min = np.abs(np.min(reconstructed) / np.min(barrier_waveform))

    return ratio_max if ratio_max > ratio_min else ratio_min
