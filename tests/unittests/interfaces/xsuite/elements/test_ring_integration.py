"""Ring/Simulation integration of the WrapXsuite4Blond black box.

The wrapper is a length-bearing, reference-altering element — deliberately
NOT a drift or RF station by type. These tests pin down that the ring's
accounting treats it accordingly: it occupies orbit length, it never counts
as an RF station (``n_rf_stations``, one-per-section check, reordering), and
it is invisible to drift analytics (``eta_0`` averaging).
"""

import warnings

import numpy as np
import pytest

xt = pytest.importorskip("xtrack")

from blond import (  # noqa: E402
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn  # noqa: E402
from blond.interfaces.xsuite.elements import WrapXsuite4Blond  # noqa: E402

CIRCUMFERENCE = 100.0


class CountingLine(xt.Line):
    """xtrack Line that counts how often ``track`` is called."""

    def __init__(self, length: float = CIRCUMFERENCE):
        super().__init__(
            elements=[xt.Drift(length=length)], element_names=["d"]
        )
        self.n_track_calls = 0

    def track(self, particles, **kwargs):
        self.n_track_calls += 1
        super().track(particles, **kwargs)


def _wrapper(length: float = CIRCUMFERENCE) -> WrapXsuite4Blond:
    line = xt.Line(elements=[xt.Drift(length=length)], element_names=["drift"])
    return WrapXsuite4Blond(line)


def _rf_station() -> SingleHarmonicRFStation:
    return SingleHarmonicRFStation(voltage=0.0, phi_rf=0.0, harmonic=10)


def _constant_cycle() -> ConstantMagneticCycle:
    return ConstantMagneticCycle(
        reference_particle=proton, value=1e9, in_unit="momentum"
    )


def test_rf_station_and_wrapper_share_a_section():
    """The wrapper's documented usage must survive the section-index check.

    ``ring.add_elements([WrapXsuite4Blond(my_line), rf, ...])`` from the
    wrapper docstring puts a real RF station and the (non-counting) wrapper
    into the same section; with default ``check_section_indices=True`` this
    must not be flagged as two RF stations in one section.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements((_rf_station(), _wrapper()), section_index=0)

    Simulation(ring=ring, magnetic_cycle=_constant_cycle())  # must not raise


def test_rf_station_and_wrapper_can_be_reordered():
    """``reorder=True`` must not count the wrapper as a second RF station."""
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements(
        (_rf_station(), _wrapper()), section_index=0, reorder=True
    )

    Simulation(ring=ring, magnetic_cycle=_constant_cycle())  # must not raise


def test_n_rf_stations_excludes_wrapper():
    """Only accelerating stations count; the black-box wrapper does not."""
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements((_rf_station(), _wrapper()), section_index=0)
    assert ring.n_rf_stations == 1


def test_guest_line_is_tracked_exactly_once_per_turn():
    """One BLonD turn = one pass through the wrapped line, no probe passes.

    A stateful guest (time-dependent variables, monitors, wakefields) must
    see exactly one ``track`` call per physical turn; any extra pass runs
    its state at the wrong rate.
    """
    from blond import Beam

    line = CountingLine()
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements((_rf_station(), WrapXsuite4Blond(line)), section_index=0)
    simulation = Simulation(ring=ring, magnetic_cycle=_constant_cycle())

    beam = Beam(intensity=1.0, particle_type=proton)
    beam.setup_beam(
        dt=np.zeros(4),
        dE=np.zeros(4),
        reference_total_energy=float(beam_total_energy()),
    )
    n_turns = 3
    simulation.run_simulation(
        beams=(beam,),
        n_turns=n_turns,
        show_progressbar=False,
        verbose=False,
    )

    assert line.n_track_calls == n_turns


def beam_total_energy() -> float:
    """Total energy matching the 1e9 eV/c proton momentum of the cycle."""
    return np.sqrt(1e9**2 + proton.mass**2)


def test_eta_analytics_ignore_the_black_box():
    """The wrapper must be invisible to drift physics (eta_0 averaging).

    A ring containing the wrapper plus a real DriftSimple carrying the
    optics parameters must support eta_0 analytics (is_below_transition,
    phi_s, BiGaussian matching all build on calc_average_eta_0); the black
    box must not poison the average.
    """

    alpha = 1e-3
    drift = DriftSimple(
        orbit_length=CIRCUMFERENCE / 2, momentum_compaction_factor=alpha
    )
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements(
        (_rf_station(), drift, _wrapper(length=CIRCUMFERENCE / 2)),
        section_index=0,
    )

    gamma = 10.0
    assert ring.calc_average_eta_0(gamma=gamma) == pytest.approx(
        drift.eta_0(gamma)
    )


def test_closed_orbit_length_includes_the_black_box():
    """The wrapper occupies length along the orbit like any ring segment."""

    drift = DriftSimple(orbit_length=30.0, momentum_compaction_factor=1e-3)
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements(
        (_rf_station(), drift, _wrapper(length=70.0)), section_index=0
    )
    assert ring.closed_orbit_length == pytest.approx(100.0)


def test_section_without_rf_station_is_legal():
    """One RF station in the entire ring is enough.

    Sections are allowed to contain no RF station (e.g. the cavity lives in
    another section, or inside a wrapped external element).
    """

    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements(
        (
            _rf_station(),
            DriftSimple(
                orbit_length=CIRCUMFERENCE / 2,
                momentum_compaction_factor=1e-3,
            ),
        ),
        section_index=0,
    )
    ring.add_elements(
        (
            DriftSimple(
                orbit_length=CIRCUMFERENCE / 2,
                momentum_compaction_factor=1e-3,
            ),
        ),
        section_index=1,
    )

    Simulation(ring=ring, magnetic_cycle=_constant_cycle())  # must not raise


def test_per_turn_cycle_without_accelerating_station_raises_clear_error():
    """A ring whose only RF-typed element is the wrapper cannot host a
    ``MagneticCyclePerTurn``: the per-turn energy increment has no BLonD-side
    station to apply it. This must fail with an explanatory error, not a bare
    ``AssertionError`` deep inside the cycle configuration.
    """
    ring = Ring(circumference=CIRCUMFERENCE)
    ring.add_elements((_wrapper(),), section_index=0)
    cycle = MagneticCyclePerTurn(
        reference_particle=proton,
        value_init=1e9,
        values_after_turn=np.full(3, 1e9),
        in_unit="momentum",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="at least one RF station"):
            Simulation(ring=ring, magnetic_cycle=cycle)
