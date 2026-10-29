"""
Resistive-wall voltage of a fixed Gaussian bunch, three ways.

The setting is that of xsuite issue #970: an electron bunch with
``sigma_z = 4.223 mm`` and ``|Q| = 1.124 nC`` in a round pipe
(``b = 13.5 mm``, ``L = 515.57 m``, ``sigma_c = 3.125e7 S/m``), binned on
``zeta in [-0.2, 0.2] m`` with 2,750, 5,500 and 11,000 slices. There, the
voltage from the point-sampled thick-wall wake grows with the slice count
(15.7, 22.3, 31.5 MV) instead of converging to the ~7.81 kV the wall
impedance predicts.

The same `ResistiveWall` is evaluated through

* ``analytic``: `PeriodicFreqSolver`, Z(f) at the exact FFT frequencies,
* ``table``: `PeriodicFreqSolver` with the Z(f) tabulated once in an
  `ImpedanceTableFreq`,
* ``pole residue``: `MultiPoleSparseSolve` with a vector fit of Z(f),

and all three must agree with each other, with an independent quadrature
of the impedance against the bunch spectrum, and must not depend on the
slice count.

Set ``DEV_DRAW=true`` in the environment to plot the comparison.
"""

import functools
import os
import warnings

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.constants import c, e
from scipy.special import erf
from skrf import Frequency, Network
from skrf.vectorFitting import VectorFitting

from blond import (
    AllowPlotting,
    Beam,
    ImpedanceTableFreq,
    ResistiveWall,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    electron,
)
from blond.core.backends.backend import backend
from blond.physics.impedances.base import (
    SupportsVectorFittedModel,
    WakeFieldSource,
)
from blond.physics.impedances.multipole.vector_fit import VectorFit
from blond.physics.impedances.solvers import (
    MultiPoleSparseSolve,
    PeriodicFreqSolver,
)
from blond.testing.backend_testing import BLonDTestCase
from blond.testing.helpers import enforce_64_bit_backend

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"

PIPE_RADIUS = 13.5e-3  # [m]
PIPE_LENGTH = 515.57  # [m]
CONDUCTIVITY = 3.125e7  # [S/m]

REFERENCE_TOTAL_ENERGY = 3e9  # [eV]
BUNCH_CHARGE = 1.124e-9  # [C]
SIGMA_DT = 0.00422280328621815 / c  # [s], sigma_z = 4.223 mm
HALF_WINDOW = 0.2 / c  # [s], zeta in [-0.2, 0.2] m
SLICE_COUNTS = (2750, 5500, 11000)
N_MACROPARTICLES = 4  # irrelevant, the profile is filled by hand

# FFT period: long enough that the wake of the periodic images is
# negligible (the wake tail falls as t**-1.5); 100 windows change the
# voltage by 3e-6 of its peak against 1000 windows. That is 1.1e6 FFT points
# at 11,000 slices, just above the solver's performance warning.
T_PERIODICITY = 100 * 2 * HALF_WINDOW  # [s]

# `ImpedanceTableFreq`: 100 MHz steps up to 200 GHz, zero above, where the
# bunch spectrum is down to exp(-2 pi^2 sigma_dt^2 f^2) ~ 1e-68.
TABLE_FREQ = np.linspace(0.0, 200e9, 2001)  # [Hz]

# Vector fit: no direct or inductive term, Z(f) vanishes at both ends.
FIT_FREQ = np.logspace(6, 13.5, 800)  # [Hz]
N_POLES_REAL = 16
N_POLES_COMPLEX = 8

# Independent reference: quadrature of Z(f) times the bunch spectrum.
QUADRATURE_FREQ = np.linspace(0.0, 100e9, 200_001)  # [Hz]
QUADRATURE_HALF_SPAN = 70e-12  # [s], where the reference is checked

# xsuite's own wall impedance gives a peak of about 7.81 kV (#970).
XSUITE_IMPEDANCE_PEAK = 7.81e3  # [V]

RTOL_TO_REFERENCE = 1e-3  # relative to the peak voltage
RTOL_BETWEEN_SLICE_COUNTS = 1e-3


def _resistive_wall() -> ResistiveWall:
    return ResistiveWall(
        pipe_radius=PIPE_RADIUS,
        pipe_length=PIPE_LENGTH,
        conductivity=CONDUCTIVITY,
    )


def _host_impedance(source: ResistiveWall, freq: np.ndarray) -> np.ndarray:
    return copy_to_cpu(
        source.get_impedance(
            backend.array(freq, dtype=backend.float),
            simulation=None,
            beam=None,
        )
    )


class _VectorFittedImpedance(WakeFieldSource, SupportsVectorFittedModel):
    """Pole-residue source holding a fixed vector fit."""

    def __init__(self, vector_fit: VectorFit):
        super().__init__(is_dynamic=False)
        self._vector_fit = vector_fit

    def get_vectorfit(self) -> VectorFit:
        return self._vector_fit


@functools.cache
def _vector_fit() -> VectorFit:
    """Vector fit of the wall impedance on `FIT_FREQ`."""
    network = Network(
        frequency=Frequency.from_f(FIT_FREQ, unit="Hz"),
        s=_host_impedance(_resistive_wall(), FIT_FREQ).reshape(-1, 1, 1),
    )
    vector_fitting = VectorFitting(network)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector_fitting.vector_fit(
            n_poles_real=N_POLES_REAL,
            n_poles_cmplx=N_POLES_COMPLEX,
            init_pole_spacing="log",
            fit_constant=False,
            fit_proportional=False,
        )
    return VectorFit(
        poles=vector_fitting.poles,
        residues=vector_fitting.residues[0],
        counterrotation_signs=np.ones(len(vector_fitting.poles)),
    )


def _make_gaussian_profile(n_slices: int) -> StaticProfile:
    """Frozen profile holding the exact Gaussian charge of each bin."""
    profile = StaticProfile(
        cut_left=-HALF_WINDOW, cut_right=HALF_WINDOW, n_bins=n_slices
    )
    hist_x = copy_to_cpu(profile.hist_x)
    edges = np.append(
        hist_x - profile.hist_step / 2, hist_x[-1] + profile.hist_step / 2
    )
    fractions = np.diff(erf(edges / (np.sqrt(2) * SIGMA_DT))) / 2
    profile.hist_y[:] = backend.array(fractions, dtype=backend.float)
    profile.hist_y_to_density_factor = 1.0
    profile.active = False  # freeze: never recomputed from the beam
    return profile


def _make_headless_beam() -> Beam:
    beam = Beam(intensity=round(BUNCH_CHARGE / e), particle_type=electron)
    beam.setup_beam(
        dt=backend.zeros(N_MACROPARTICLES, dtype=backend.float),
        dE=backend.zeros(N_MACROPARTICLES, dtype=backend.float),
        reference_time=0.0,
        reference_total_energy=REFERENCE_TOTAL_ENERGY,
    )
    return beam


@functools.cache
def _induced_voltages(n_slices: int) -> tuple[np.ndarray, dict]:
    """
    Induced voltage per route for one slice count.

    Parameters
    ----------
    n_slices
        Number of profile bins.

    Returns
    -------
    hist_x
        Bin centres, in [s].
    voltages
        Induced voltage, in [V], per route name.
    """
    resistive_wall = _resistive_wall()
    routes = {
        "analytic": (
            resistive_wall,
            PeriodicFreqSolver(
                t_periodicity=T_PERIODICITY, warn_above_n_time=None
            ),
        ),
        "table": (
            ImpedanceTableFreq(
                freq_x=TABLE_FREQ,
                freq_y=_host_impedance(resistive_wall, TABLE_FREQ),
            ),
            PeriodicFreqSolver(
                t_periodicity=T_PERIODICITY, warn_above_n_time=None
            ),
        ),
        "pole residue": (
            _VectorFittedImpedance(_vector_fit()),
            MultiPoleSparseSolve(),
        ),
    }
    voltages = {}
    for name, (source, solver) in routes.items():
        profile = _make_gaussian_profile(n_slices)
        beam = _make_headless_beam()
        wakefield = WakeField.headless(
            beam=beam, sources=(source,), solver=solver, profile=profile
        )
        voltages[name] = copy_to_cpu(wakefield.calc_induced_voltage(beam))
    return copy_to_cpu(profile.hist_x), voltages


def _quadrature_voltage(times: np.ndarray) -> np.ndarray:
    """
    Induced voltage from Z(f) and the analytic bunch spectrum, in [V].

    ``V(t) = -q N int Z(f) Lambda(f) exp(2 pi i f t) df`` over all f, with
    the Gaussian spectrum ``Lambda(f) = exp(-2 pi^2 sigma_dt^2 f^2)`` and
    ``Z(-f) = conj(Z(f))``; BLonD's Fourier sign convention.
    """
    impedance = _host_impedance(_resistive_wall(), QUADRATURE_FREQ)
    spectrum = np.exp(-2 * np.pi**2 * SIGMA_DT**2 * QUADRATURE_FREQ**2)
    integrand = (impedance * spectrum)[np.newaxis, :] * np.exp(
        2j * np.pi * QUADRATURE_FREQ[np.newaxis, :] * times[:, np.newaxis]
    )
    bunch_charge = electron.charge * e * round(BUNCH_CHARGE / e)
    return -bunch_charge * 2 * np.trapezoid(integrand, QUADRATURE_FREQ).real


def _relative_deviation(voltage, reference) -> float:
    return float(
        np.max(np.abs(voltage - reference)) / np.max(np.abs(reference))
    )


@pytest.mark.integration
class TestResistiveWallGaussianBunch(BLonDTestCase):
    """All routes agree and converge for the #970 bunch."""

    def setUp(self):
        enforce_64_bit_backend()

    def test_analytic_matches_quadrature(self):
        for n_slices in SLICE_COUNTS:
            with self.subTest(n_slices=n_slices):
                hist_x, voltages = _induced_voltages(n_slices)
                near = np.abs(hist_x) < QUADRATURE_HALF_SPAN
                reference = _quadrature_voltage(hist_x[near][::10])
                self.assertLess(
                    _relative_deviation(
                        voltages["analytic"][near][::10], reference
                    ),
                    RTOL_TO_REFERENCE,
                )

    def test_routes_agree(self):
        for n_slices in SLICE_COUNTS:
            _, voltages = _induced_voltages(n_slices)
            for name in ("table", "pole residue"):
                with self.subTest(n_slices=n_slices, route=name):
                    self.assertLess(
                        _relative_deviation(
                            voltages[name], voltages["analytic"]
                        ),
                        RTOL_TO_REFERENCE,
                    )

    def test_independent_of_slice_count(self):
        # The failure of #970: the voltage grew with the slice count.
        peaks = {
            n_slices: {
                name: np.max(np.abs(voltage))
                for name, voltage in _induced_voltages(n_slices)[1].items()
            }
            for n_slices in SLICE_COUNTS
        }
        finest = peaks[SLICE_COUNTS[-1]]
        for n_slices in SLICE_COUNTS[:-1]:
            for name, peak in peaks[n_slices].items():
                with self.subTest(n_slices=n_slices, route=name):
                    self.assertLess(
                        abs(peak / finest[name] - 1),
                        RTOL_BETWEEN_SLICE_COUNTS,
                    )

    def test_peak_matches_xsuite_impedance_reference(self):
        _, voltages = _induced_voltages(SLICE_COUNTS[-1])
        self.assertAlmostEqual(
            float(np.max(voltages["analytic"])) / XSUITE_IMPEDANCE_PEAK,
            1.0,
            delta=RTOL_TO_REFERENCE,
        )

    @pytest.mark.skipif(not _DEV_DRAW, reason="set DEV_DRAW=true to plot")
    def test_draw(self):  # pragma: no cover
        _draw()


def _draw() -> None:  # pragma: no cover
    """Profile (top) and induced voltage (bottom) per slice count."""
    if matplotlib.get_backend().lower().endswith("agg"):
        for backend_name in ("qtagg", "tkagg", "gtk4agg", "gtk3agg"):
            try:
                matplotlib.use(backend_name, force=True)
                break
            except Exception:  # backend not installed, try the next one
                continue
    styles = {
        "analytic": dict(color="C0", lw=3, alpha=0.5),
        "table": dict(color="C1", lw=1.5),
        "pole residue": dict(color="k", lw=1, ls="--"),
    }
    with AllowPlotting():
        fig, axes = plt.subplots(
            2,
            len(SLICE_COUNTS),
            sharex=True,
            sharey="row",
            figsize=(15, 7),
            gridspec_kw=dict(height_ratios=(1, 2)),
        )
        for column, n_slices in enumerate(SLICE_COUNTS):
            hist_x, voltages = _induced_voltages(n_slices)
            profile = _make_gaussian_profile(n_slices)
            bin_dt = profile.hist_step
            ax_profile, ax_voltage = axes[:, column]
            ax_profile.bar(
                hist_x * 1e12,
                copy_to_cpu(profile.hist_y) * BUNCH_CHARGE * 1e12,
                width=bin_dt * 1e12,
                color="C2",
            )
            ax_profile.set_title(
                f"{n_slices:,} slices, dt = {bin_dt * 1e12:.3f} ps"
            )
            for name, voltage in voltages.items():
                ax_voltage.plot(
                    hist_x * 1e12, voltage * 1e-3, label=name, **styles[name]
                )
            ax_voltage.set_xlabel("dt relative to bunch centre [ps]")
        axes[0, 0].set_ylabel("charge per bin [pC]")
        axes[1, 0].set_ylabel("induced voltage [kV]")
        axes[1, 0].legend()
        axes[1, 0].set_xlim(
            -QUADRATURE_HALF_SPAN * 1e12, QUADRATURE_HALF_SPAN * 1e12
        )
        fig.suptitle("Resistive wall, fixed Gaussian bunch (xsuite #970)")
        fig.tight_layout()
        plt.show(block=True)
