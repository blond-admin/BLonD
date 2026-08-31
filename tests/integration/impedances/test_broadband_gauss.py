"""
Same solver comparison as `broadband.py`, but with a Gaussian bunch.

Instead of the spike comb of `broadband.py`, the profile holds a single
Gaussian line density centred in the profile window with an rms length of
`SIGMA_DT`. Everything else is unchanged: one resonator at `f_res`, and a
frequency-domain, a time-domain and a pole-residue solver evaluated on the
identical frozen profile.

Two frequencies matter:

* ``f_res`` -- the resonator centre frequency.
* ``f_cutoff`` -- the Nyquist frequency of the profile binning
  (`StaticProfile.cutoff_frequency`, ``1 / (2 * hist_step)``).

The two cases keep the same bunch and resonator and only change the
binning: ``f_res = 10 * f_cutoff`` (resonator far above the profile's
Nyquist frequency) and ``f_cutoff = 3 * f_res`` (resolved reference).

Note that a Gaussian of rms length `SIGMA_DT` has essentially no spectral
content at `f_res`: its spectrum falls off as
``exp(-2 * pi**2 * sigma_dt**2 * f**2)``, i.e. it is already negligible
above ``1 / (2 * pi * sigma_dt)``. The induced voltage is therefore driven
by the low-frequency (inductive) flank of the resonator, not by its peak.

This is an investigation script, not a unittest: run it directly.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    Numpy64Bit,
    Resonators,
    StaticProfile,
    WakeField,
    proton,
)
from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu
from blond.physics.impedances.solvers import (
    MultiPoleSparseSolve,
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)

F_RES = 1e9  # resonator centre frequency, in [Hz]
R_SHUNT = 1e6  # shunt impedance, in [Ohm]
AMPLITUDE_HALF_TIME = 0.2e-9  # wake amplitude halving time, in [s]

SIGMA_DT = 12e-9  # rms bunch length, in [s]
WINDOW_LENGTH = 160e-9  # profile window, in [s], as in `broadband.py`
BUNCH_CENTER = 0.5 * WINDOW_LENGTH  # in [s]

# The amplitude decay over a time `dt` is ``exp(-pi * f_res * dt / Q)``.
QUALITY_FACTOR = np.pi * F_RES * AMPLITUDE_HALF_TIME / np.log(2.0)

BEAM_INTENSITY = 1e11
REFERENCE_TOTAL_ENERGY = 450e9  # in [eV], only sets the beam reference frame
N_MACROPARTICLES = 4  # irrelevant, the profile is filled by hand


@dataclass(frozen=True)
class Case:
    """
    One binning of the same Gaussian bunch and resonator.

    Attributes
    ----------
    label
        Short description used in the figure title.
    cutoff_frequency
        Nyquist frequency of the profile binning, in [Hz].
    """

    label: str
    cutoff_frequency: float


CASES = (
    Case(
        label="f_res = 10 * f_cutoff",
        cutoff_frequency=F_RES / 10.0,
    ),
    Case(
        label="f_cutoff = 3 * f_res",
        cutoff_frequency=3.0 * F_RES,
    ),
)


def make_gaussian_profile(case: Case) -> StaticProfile:
    """
    Static profile holding a centred Gaussian bunch, frozen for tracking.

    Parameters
    ----------
    case
        Binning to build the profile for.

    Returns
    -------
    profile
        Profile with a Gaussian `hist_y`, `active=False`.
    """
    profile = StaticProfile.from_cutoff(
        cut_left=0.0,
        cut_right=WINDOW_LENGTH,
        cutoff_frequency=case.cutoff_frequency,
    )
    hist_x = copy_to_cpu(profile.hist_x)
    line_density = np.exp(
        -0.5 * ((hist_x - BUNCH_CENTER) / SIGMA_DT) ** 2,
    )

    profile.hist_y[:] = backend.array(line_density, dtype=backend.float)
    # Normalise so that `hist_y * hist_y_to_density_factor` sums to one,
    # i.e. the bunch carries exactly `BEAM_INTENSITY` charges.
    profile.hist_y_to_density_factor = 1.0 / float(np.sum(line_density))
    profile.active = False  # freeze: never recomputed from the beam
    profile.invalidate_cache()
    return profile


def make_headless_beam() -> Beam:
    """
    Beam that carries only intensity, particle type and a reference time.

    The macroparticle coordinates are irrelevant: the profile is frozen and
    filled by hand, so the beam is never binned.

    Returns
    -------
    beam
        Beam ready to be passed to a solver outside a `Simulation`.
    """
    beam = Beam(intensity=BEAM_INTENSITY, particle_type=proton)
    beam.setup_beam(
        dt=backend.zeros(N_MACROPARTICLES, dtype=backend.float),
        dE=backend.zeros(N_MACROPARTICLES, dtype=backend.float),
        reference_time=0.0,
        reference_total_energy=REFERENCE_TOTAL_ENERGY,
    )
    return beam


def induced_voltages(profile: StaticProfile) -> dict[str, np.ndarray]:
    """
    Induced voltage of one resonator for each solver, in a single pass.

    Parameters
    ----------
    profile
        Frozen beam profile driving the resonator.

    Returns
    -------
    voltages
        Induced voltage in [V] per solver name.
    """
    solvers = {
        "frequency domain": PeriodicFreqSolver(
            t_periodicity=WINDOW_LENGTH,
        ),
        "time domain": TimeDomainFftSolver(),
        "pole residue": MultiPoleSparseSolve(),
    }
    voltages = {}
    for name, solver in solvers.items():
        beam = make_headless_beam()
        wakefield = WakeField.headless(
            beam=beam,
            sources=(
                Resonators(
                    shunt_impedances=R_SHUNT,
                    center_frequencies=F_RES,
                    quality_factors=QUALITY_FACTOR,
                ),
            ),
            solver=solver,
            profile=profile,
        )
        voltages[name] = copy_to_cpu(wakefield.calc_induced_voltage(beam=beam))
    return voltages


def plot_case(case: Case) -> None:
    """
    Plot profile, induced voltages and profile spectrum for one binning.

    Parameters
    ----------
    case
        Binning to run.
    """
    profile = make_gaussian_profile(case)
    voltages = induced_voltages(profile)

    hist_x = copy_to_cpu(profile.hist_x)
    fig, (ax_profile, ax_voltage, ax_spectrum) = plt.subplots(
        3, 1, figsize=(8, 9)
    )
    ax_voltage.sharex(ax_profile)
    fig.suptitle(
        f"{case.label}\n"
        f"f_res = {F_RES / 1e9:.3f} GHz, "
        f"f_cutoff = {case.cutoff_frequency / 1e9:.3f} GHz, "
        f"sigma_dt = {SIGMA_DT * 1e9:.1f} ns, "
        f"Q = {QUALITY_FACTOR:.1f}, {profile.n_bins} bins"
    )
    ax_profile.plot(hist_x * 1e9, copy_to_cpu(profile.hist_y), "o-", ms=3)
    ax_profile.set_ylabel("profile [a.u.]")

    print(f"--- {case.label}")
    for name, voltage in voltages.items():
        print(f"    {name:18s} peak |V| = {np.max(np.abs(voltage)):.3e} V")
        ax_voltage.plot(hist_x * 1e9, voltage, label=name)
    ax_voltage.set_ylabel("induced voltage [V]")
    ax_voltage.set_xlabel("time [ns]")
    ax_voltage.legend()

    # Spectrum of the profile as the solvers see it: the rfft stops at the
    # profile cutoff frequency, so a resonator beyond it is never sampled.
    spectrum = copy_to_cpu(profile.beam_spectrum(n_fft=profile.n_bins))
    spectrum_freq = np.fft.rfftfreq(profile.n_bins, d=profile.hist_step)
    spectrum_style = "o-" if profile.n_bins <= 128 else "-"
    ax_spectrum.semilogy(
        spectrum_freq / 1e9, np.abs(spectrum), spectrum_style, ms=3, lw=0.8
    )
    ax_spectrum.axvline(
        F_RES / 1e9, color="k", ls="--", label=f"f_res = {F_RES / 1e9:.3f} GHz"
    )
    ax_spectrum.axvline(
        case.cutoff_frequency / 1e9,
        color="r",
        ls=":",
        label=f"f_cutoff = {case.cutoff_frequency / 1e9:.3f} GHz",
    )
    ax_spectrum.set_xlim(0.0, 1.2 * max(F_RES, case.cutoff_frequency) / 1e9)
    ax_spectrum.set_ylim(bottom=1e-12 * float(np.max(np.abs(spectrum))))
    ax_spectrum.set_ylabel("|profile spectrum| [a.u.]")
    ax_spectrum.set_xlabel("frequency [GHz]")
    ax_spectrum.legend()
    fig.tight_layout()


def print_convergence_scan(
    cutoff_ratios: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
) -> None:
    """
    Print the peak induced voltage per solver over a range of binnings.

    The bunch and the resonator are identical for every entry, so any
    change of the peak voltage is a pure discretisation effect.

    Parameters
    ----------
    cutoff_ratios
        Profile cutoff frequencies to scan, in units of `F_RES`.
    """
    header = f"{'f_cutoff/f_res':>14} {'n_bins':>7}"
    print(f"{header} {'freq':>12} {'time':>12} {'pole':>12}")
    for ratio in cutoff_ratios:
        case = Case(label=f"{ratio}", cutoff_frequency=ratio * F_RES)
        profile = make_gaussian_profile(case)
        peaks = [
            np.max(np.abs(voltage))
            for voltage in induced_voltages(profile).values()
        ]
        peaks_str = " ".join(f"{peak:12.3e}" for peak in peaks)
        print(f"{ratio:14.1f} {profile.n_bins:7d} {peaks_str}")


if __name__ == "__main__":
    backend.change_backend(Numpy64Bit)
    backend.set_specials("numba")

    for case_ in CASES:
        plot_case(case_)
    print_convergence_scan()
    plt.show()
