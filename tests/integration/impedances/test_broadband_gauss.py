"""
Solver agreement for a Gaussian bunch driving a broadband resonator.

A single Gaussian bunch of rms length `SIGMA_DT` sits in the middle of a
frozen `StaticProfile`; one broadband resonator (`f_res`, near-critically
damped) is evaluated on that profile with three solvers:
`PeriodicFreqSolver` (frequency domain), `TimeDomainFftSolver` (time
domain) and `MultiPoleSparseSolve` (pole residue).

Two frequencies matter:

* ``f_res`` -- the resonator centre frequency.
* ``f_cutoff`` -- the Nyquist frequency of the profile binning
  (`StaticProfile.cutoff_frequency`, ``1 / (2 * hist_step)``).

The bunch has no spectral content anywhere near `f_res` (a Gaussian
spectrum falls off as ``exp(-2 * pi**2 * sigma_dt**2 * f**2)``, i.e. it is
already negligible above ``1 / (2 * pi * sigma_dt)`` = 13 MHz here), so the
induced voltage is driven purely by the low-frequency inductive flank of
the resonator. The physical answer therefore does not depend on the
binning at all, which is what these tests exploit: the frequency-domain
solver returns the same voltage for every binning tested here, and the
time-domain solvers must converge onto it.

They now do at every binning that resolves the bunch -- see
`TestResonatorAboveProfileCutoff`, which pins the case that used to fail.

Set ``DEV_DRAW=true`` in the environment to plot the comparison.
"""

import os
import unittest

import numpy as np
import pytest
from matplotlib import pyplot as plt

from blond import (
    AllowPlotting,
    Beam,
    Resonators,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    proton,
)
from blond.core.backends.backend import backend
from blond.physics.impedances.solvers import (
    MultiPoleSparseSolve,
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)
from blond.testing.helpers import enforce_64_bit_backend

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"

F_RES = 1e9  # resonator centre frequency, in [Hz]
R_SHUNT = 1e6  # shunt impedance, in [Ohm]
AMPLITUDE_HALF_TIME = 0.2e-9  # wake amplitude halving time, in [s]

SIGMA_DT = 12e-9  # rms bunch length, in [s]
WINDOW_LENGTH = 160e-9  # profile window, in [s]
BUNCH_CENTER = 0.5 * WINDOW_LENGTH  # in [s]

# The wake amplitude decay over a time `dt` is ``exp(-pi * f_res * dt / Q)``,
# which gives a near-critically damped (broadband) resonator here.
QUALITY_FACTOR = np.pi * F_RES * AMPLITUDE_HALF_TIME / np.log(2.0)

BEAM_INTENSITY = 1e11
REFERENCE_TOTAL_ENERGY = 450e9  # in [eV], only sets the beam reference frame
N_MACROPARTICLES = 4  # irrelevant, the profile is filled by hand

# Binnings that resolve the bunch itself (bin width well below `SIGMA_DT`),
# expressed as `f_cutoff / f_res`.
BUNCH_RESOLVING_CUTOFF_RATIOS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
# Binning that also resolves the wake (bin width well below the wake's
# `AMPLITUDE_HALF_TIME`).
WAKE_RESOLVING_CUTOFF_RATIO = 30.0
# Binning of interest: the resonator sits a decade above the profile cutoff.
RESONATOR_ABOVE_CUTOFF_RATIO = 0.1

SOLVER_AGREEMENT_RTOL = 0.1


def _make_gaussian_profile(cutoff_frequency: float) -> StaticProfile:
    """
    Static profile holding a centred Gaussian bunch, frozen for tracking.

    Parameters
    ----------
    cutoff_frequency
        Nyquist frequency of the profile binning, in [Hz].

    Returns
    -------
    profile
        Profile with a Gaussian `hist_y`, `active=False`.
    """
    profile = StaticProfile.from_cutoff(
        cut_left=0.0,
        cut_right=WINDOW_LENGTH,
        cutoff_frequency=cutoff_frequency,
    )
    hist_x = np.asarray(copy_to_cpu(profile.hist_x))
    line_density = np.exp(-0.5 * ((hist_x - BUNCH_CENTER) / SIGMA_DT) ** 2)

    profile.hist_y[:] = backend.array(line_density, dtype=backend.float)
    # Normalise so that `hist_y * hist_y_to_density_factor` sums to one,
    # i.e. the bunch carries exactly `BEAM_INTENSITY` charges independently
    # of the binning.
    profile.hist_y_to_density_factor = 1.0 / float(np.sum(line_density))
    profile.active = False  # freeze: never recomputed from the beam
    profile.invalidate_cache()
    return profile


def _make_headless_beam() -> Beam:
    """
    Beam carrying only intensity, particle type and a reference time.

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


def _peak_voltages(
    cutoff_frequency: float,
    solver_names: tuple[str, ...] | None = None,
    center_frequency: float = F_RES,
) -> dict[str, float]:
    """
    Peak absolute induced voltage per solver for one profile binning.

    Parameters
    ----------
    cutoff_frequency
        Nyquist frequency of the profile binning, in [Hz].
    solver_names
        Solvers to evaluate; all of them when ``None``.
    center_frequency
        Resonator centre frequency, in [Hz].

    Returns
    -------
    peak_voltages
        Peak absolute induced voltage, in [V], per solver name.
    """
    return {
        name: float(np.max(np.abs(voltage)))
        for name, voltage in _induced_voltages(
            cutoff_frequency, solver_names, center_frequency
        )[1].items()
    }


def _induced_voltages(
    cutoff_frequency: float,
    solver_names: tuple[str, ...] | None = None,
    center_frequency: float = F_RES,
) -> tuple[StaticProfile, dict[str, np.ndarray]]:
    """
    Induced voltage of the resonator for each solver, in a single pass.

    Parameters
    ----------
    cutoff_frequency
        Nyquist frequency of the profile binning, in [Hz].
    solver_names
        Solvers to evaluate; all of them when ``None``.
    center_frequency
        Resonator centre frequency, in [Hz].

    Returns
    -------
    profile
        The frozen profile the solvers were evaluated on.
    voltages
        Induced voltage in [V] per solver name.
    """
    profile = _make_gaussian_profile(cutoff_frequency)
    solvers = {
        "frequency domain": PeriodicFreqSolver(t_periodicity=WINDOW_LENGTH),
        "time domain": TimeDomainFftSolver(),
        "pole residue": MultiPoleSparseSolve(),
    }
    if solver_names is not None:
        solvers = {name: solvers[name] for name in solver_names}
    voltages = {}
    for name, solver in solvers.items():
        beam = _make_headless_beam()
        wakefield = WakeField.headless(
            beam=beam,
            sources=(
                Resonators(
                    shunt_impedances=R_SHUNT,
                    center_frequencies=center_frequency,
                    quality_factors=QUALITY_FACTOR,
                ),
            ),
            solver=solver,
            profile=profile,
        )
        voltages[name] = np.asarray(
            copy_to_cpu(wakefield.calc_induced_voltage(beam=beam))
        )
    return profile, voltages


def _draw(cutoff_ratio: float) -> None:
    """
    Plot profile and induced voltages for one binning, for `DEV_DRAW`.

    Parameters
    ----------
    cutoff_ratio
        Profile cutoff frequency, in units of `F_RES`.
    """
    profile, voltages = _induced_voltages(cutoff_ratio * F_RES)
    hist_x = np.asarray(copy_to_cpu(profile.hist_x))
    with AllowPlotting():
        fig, (ax_profile, ax_voltage) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6)
        )
        fig.suptitle(
            f"f_cutoff = {cutoff_ratio:g} * f_res, "
            f"sigma_dt = {SIGMA_DT * 1e9:.1f} ns, "
            f"Q = {QUALITY_FACTOR:.2f}, {profile.n_bins} bins"
        )
        ax_profile.plot(hist_x * 1e9, copy_to_cpu(profile.hist_y))
        ax_profile.set_ylabel("profile [a.u.]")
        for name, voltage in voltages.items():
            ax_voltage.plot(hist_x * 1e9, voltage, label=name)
        ax_voltage.set_ylabel("induced voltage [V]")
        ax_voltage.set_xlabel("time [ns]")
        ax_voltage.legend()
        fig.tight_layout()
        plt.show()


@pytest.mark.integration
class TestFrequencyDomainReference(unittest.TestCase):
    """The frequency-domain solver is the binning-independent reference."""

    def setUp(self):
        enforce_64_bit_backend()

    def test_frequency_domain_is_independent_of_binning(self):
        """
        Refining the binning must not change the frequency-domain voltage.

        The bunch spectrum dies far below `f_res`, so once the binning
        resolves the bunch there is nothing left for a finer grid to add.
        """
        peaks = [
            _peak_voltages(ratio * F_RES)["frequency domain"]
            for ratio in BUNCH_RESOLVING_CUTOFF_RATIOS
        ]
        for ratio, peak in zip(
            BUNCH_RESOLVING_CUTOFF_RATIOS[1:], peaks[1:], strict=True
        ):
            self.assertAlmostEqual(
                peak / peaks[0],
                1.0,
                delta=SOLVER_AGREEMENT_RTOL,
                msg=(
                    f"frequency-domain peak voltage changed by "
                    f"{abs(peak / peaks[0] - 1.0):.1%} when going from "
                    f"f_cutoff = {BUNCH_RESOLVING_CUTOFF_RATIOS[0]} * f_res "
                    f"to {ratio} * f_res, but the physical voltage cannot "
                    f"depend on the binning"
                ),
            )


@pytest.mark.integration
class TestWakeResolvedBinning(unittest.TestCase):
    """All three solvers agree once the binning resolves the wake."""

    def setUp(self):
        enforce_64_bit_backend()

    def test_all_solvers_agree(self):
        """
        Time-domain and pole-residue must match the frequency domain.

        With ``f_cutoff = 30 * f_res`` the bin width is far below the
        wake's amplitude halving time, so every solver sees the same wake.
        """
        peaks = _peak_voltages(WAKE_RESOLVING_CUTOFF_RATIO * F_RES)
        reference = peaks["frequency domain"]
        for name in ("time domain", "pole residue"):
            self.assertAlmostEqual(
                peaks[name] / reference,
                1.0,
                delta=SOLVER_AGREEMENT_RTOL,
                msg=(
                    f"{name} solver gives {peaks[name]:.3e} V, "
                    f"frequency domain gives {reference:.3e} V"
                ),
            )

    def test_time_domain_and_pole_residue_are_equivalent(self):
        """The two time-domain formulations must give the same voltage."""
        peaks = _peak_voltages(WAKE_RESOLVING_CUTOFF_RATIO * F_RES)
        self.assertAlmostEqual(
            peaks["pole residue"] / peaks["time domain"],
            1.0,
            delta=1e-3,
            msg=(
                f"pole residue gives {peaks['pole residue']:.3e} V, "
                f"time domain gives {peaks['time domain']:.3e} V"
            ),
        )


@pytest.mark.integration
class TestResonatorAboveProfileCutoff(unittest.TestCase):
    """
    Binning that resolves the bunch but not the wake.

    Note
    ----
    REGRESSION GUARD: with ``f_res = 10 * f_cutoff`` the bin width (5 ns)
    is far above the wake's amplitude halving time (0.2 ns), so
    `TimeDomainFftSolver` and `MultiPoleSparseSolve` sample a wake they
    cannot represent. The bunch itself is well resolved and its spectrum
    dies decades below `f_res`, so the physical voltage is the
    binning-independent one that `PeriodicFreqSolver` returns -- and both
    time-domain solvers must return it here too, without the binning ever
    resolving the wake.

    They used to return about 0.3 % of it (low by a factor of ~300).
    Bin-averaging the wake over the source bin alone (a box,
    ``sinc(f dt)``; for the poles ``(exp(p * dt) - 1) / (p * dt)``) does
    not rescue this: sampling folds the resonance from above the Nyquist
    frequency down onto the impedance's inductive low-frequency flank --
    the only part of the impedance this bunch samples -- and a single box
    suppresses that fold only to first order in ``f * dt``, so it very
    nearly cancels the flank. Averaging over the observation bin as well
    (the triangle, ``sinc(f dt)**2``) suppresses it to second order and
    leaves the flank intact; see `Resonators._wake_bin_average`.
    """

    def setUp(self):
        enforce_64_bit_backend()

    def test_time_domain_and_pole_residue_are_equivalent(self):
        """
        The two time-domain formulations fail in the same way.

        Whatever the binning does to them, it does identically -- so the
        discrepancy below is not specific to the pole-residue recursion.
        """
        peaks = _peak_voltages(RESONATOR_ABOVE_CUTOFF_RATIO * F_RES)
        self.assertAlmostEqual(
            peaks["pole residue"] / peaks["time domain"],
            1.0,
            delta=1e-3,
            msg=(
                f"pole residue gives {peaks['pole residue']:.3e} V, "
                f"time domain gives {peaks['time domain']:.3e} V"
            ),
        )

    def test_all_solvers_agree(self):
        """
        Time-domain and pole-residue must match the frequency domain.

        Both used to return ~0.3 % of the correct induced voltage -- this
        is the case this module is about, see the class docstring.
        """
        peaks = _peak_voltages(RESONATOR_ABOVE_CUTOFF_RATIO * F_RES)
        reference = peaks["frequency domain"]
        for name in ("time domain", "pole residue"):
            self.assertAlmostEqual(
                peaks[name] / reference,
                1.0,
                delta=SOLVER_AGREEMENT_RTOL,
                msg=(
                    f"{name} solver gives {peaks[name]:.3e} V, "
                    f"frequency domain gives {reference:.3e} V "
                    f"({peaks[name] / reference:.1f}x)"
                ),
            )


@pytest.mark.integration
class TestExtremelyUnderResolvedPole(unittest.TestCase):
    """Solvers stay exact when the wake dies well inside a single bin."""

    def setUp(self):
        enforce_64_bit_backend()

    def test_all_solvers_agree_far_past_the_cutoff(self):
        """
        Every solver must hold up decades past the profile cutoff.

        For a resonator 100x above `F_RES` on the same 5 ns binning the wake
        decays by ``exp(-1733)`` within one bin. Both time-domain solvers
        used to break there: the convolution ones would evaluate
        ``sinh(p * dt / 2)**2 * exp(p * t)`` as ``inf * 0 = nan``, and the
        pole recursion cancelled a residue scaled by ``exp(-p * dt)`` against
        its self-bin term, losing the whole mantissa. Both now keep every
        factor bounded by one -- see `Resonators._wake_bin_average` and
        `MultiPoleSparseSolve._finalize_solver`.
        """
        peaks = _peak_voltages(
            RESONATOR_ABOVE_CUTOFF_RATIO * F_RES,
            center_frequency=100.0 * F_RES,
        )
        reference = peaks["frequency domain"]
        for name in ("time domain", "pole residue"):
            self.assertTrue(
                np.isfinite(peaks[name]), msg=f"{name} returned {peaks[name]}"
            )
            self.assertAlmostEqual(
                peaks[name] / reference,
                1.0,
                delta=SOLVER_AGREEMENT_RTOL,
                msg=(
                    f"{name} solver gives {peaks[name]:.3e} V, "
                    f"frequency domain gives {reference:.3e} V"
                ),
            )

    def test_pole_residue_tracks_time_domain_across_binnings(self):
        """
        The recursion must match the convolution at every resolution.

        Sweeps the resonator from a decade below the profile cutoff to four
        decades above it, i.e. ``-Re(p) * dt`` from 0.017 to 1.7e4. The old
        residue scaling lost ``~eps * exp(-Re(p) * dt)`` of the mantissa over
        that range (1e7 relative error at 52); the two must now agree to
        round-off throughout.
        """
        for center_frequency_ratio in (1e-3, 1e-1, 1.0, 10.0, 100.0, 1000.0):
            with self.subTest(f_res=center_frequency_ratio * F_RES):
                peaks = _peak_voltages(
                    RESONATOR_ABOVE_CUTOFF_RATIO * F_RES,
                    ("time domain", "pole residue"),
                    center_frequency=center_frequency_ratio * F_RES,
                )
                self.assertAlmostEqual(
                    peaks["pole residue"] / peaks["time domain"],
                    1.0,
                    delta=1e-9,
                    msg=(
                        f"pole residue gives {peaks['pole residue']:.6e} V, "
                        f"time domain gives {peaks['time domain']:.6e} V"
                    ),
                )


if __name__ == "__main__" and _DEV_DRAW:
    unittest.main()
