# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Unit tests for the portable cycle definition models."""

import math
import warnings

import numpy as np
import pytest
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    interp1d,
)

from blond.acc_math.analytic.simple_math import calc_total_energy
from blond.core.base import BeamPhysicsRelevant
from blond.core.beam.particle_types import ParticleType, electron, proton
from blond.core.ring.ring import Ring
from blond.core.scheduling import ScheduledArray, ScheduledInterpolation
from blond.cycles.magnetic_cycle import (
    ConstantMagneticCycle,
    MagneticCycleByTime,
    MagneticCyclePerTurn,
)
from blond.generals.interpolators import DerivativeInterpolator
from blond.physics.drifts import DriftExact, DriftSimple, _DriftLikeLineSegment
from blond.physics.rf_station import (
    MultiHarmonicRFStation,
    SingleHarmonicRFStation,
)
from blond.testing.backend_testing import BLonDTestCase

pytest.importorskip("pydantic")
import pydantic as pyd

from blond.interfaces.portable import (  # noqa: E402
    SCHEMA_VERSION,
    CycleDefinition,
    DriftDefinition,
    MagneticProgramme,
    ParticleDefinition,
    Programme,
    RFHarmonicDefinition,
    RFStationDefinition,
)


def _programme(value=(1.0, 2.0), **kwargs):
    return Programme(time=tuple(range(len(value))), value=value, **kwargs)


def _particle(**kwargs):
    return ParticleDefinition(**{"mass": 938.272e6, "charge": 1, **kwargs})


def _magnetic(**kwargs):
    return MagneticProgramme(**{"particle": _particle(), **kwargs})


def _constant(value):
    return Programme(time=(0.0,), value=(value,))


def _harmonic(harmonic=1.0, voltage=1e6, phi_rf_design=0.0):
    return RFHarmonicDefinition(
        harmonic=_constant(harmonic),
        voltage=_constant(voltage),
        phi_rf_design=_constant(phi_rf_design),
    )


def _ramp(time, value, **kwargs):
    return Programme(time=time, value=value, **kwargs)


def _harmonic_of(**programmes):
    defaults = {
        "harmonic": _constant(1.0),
        "voltage": _constant(1e6),
        "phi_rf_design": _constant(0.0),
    }
    return RFHarmonicDefinition(**{**defaults, **programmes})


class _OtherElement(BeamPhysicsRelevant):
    """Element the portable model does not represent."""

    def _track(self, beam):
        pass

    def on_init_simulation(self, simulation, **kwargs):
        pass

    def on_run_simulation(self, simulation, beam, n_turns, **kwargs):
        pass


def _station(section_index=0, name=None):
    return RFStationDefinition(
        harmonics=(_harmonic(),), section_index=section_index, name=name
    )


def _drift(**kwargs):
    return DriftDefinition(
        **{
            "orbit_length": 100.0,
            "momentum_compaction_factor": _constant(0.1),
            **kwargs,
        }
    )


def _two_section_cycle(**kwargs):
    return _cycle(
        rf_stations=(_station(0, "rf0"), _station(1, "rf1")),
        drifts=(
            _drift(orbit_length=60.0, section_index=0, name="d0"),
            _drift(orbit_length=40.0, section_index=1, name="d1"),
        ),
        **kwargs,
    )


def _cycle(**kwargs):
    arguments = {
        "circumference": 100.0,
        "magnetic_cycle": _magnetic(time=(0.0, 1.0), value=(1e9, 2e9)),
        "rf_stations": (RFStationDefinition(harmonics=(_harmonic(),)),),
        "drifts": (
            DriftDefinition(
                orbit_length=100.0, momentum_compaction_factor=_constant(0.1)
            ),
        ),
        **kwargs,
    }
    return CycleDefinition(**arguments)


class TestProgramme(BLonDTestCase):
    def test_is_constant(self):
        self.assertTrue(_constant(1.0).is_constant)
        self.assertFalse(_ramp((0.0, 1.0), (1.0, 1.0)).is_constant)

    def test_is_constant_inherited(self):
        self.assertTrue(_magnetic(time=(0.0,), value=(1e9,)).is_constant)

    def test_is_constant_not_serialised(self):
        programme = _constant(1.0)
        self.assertNotIn("is_constant", programme.model_dump())
        self.assertEqual(
            Programme.model_validate_json(programme.model_dump_json()),
            programme,
        )

    def test_defaults(self):
        programme = _programme()
        self.assertEqual(programme.interpolator, "interp1d")
        self.assertEqual(programme.interpolator_kwargs, {})

    def test_single_point_is_valid(self):
        programme = _constant(3.0)
        self.assertEqual(programme.time, (0.0,))
        self.assertEqual(programme.value, (3.0,))

    def test_arrays_are_stored_as_tuples(self):
        programme = Programme(
            time=np.array([0.0, 1.0]), value=np.array([5.0, 6.0])
        )
        self.assertEqual(programme.time, (0.0, 1.0))
        self.assertEqual(programme.value, (5.0, 6.0))

    def test_empty_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            Programme(time=(), value=())

    def test_length_mismatch_rejected(self):
        with self.assertRaisesRegex(pyd.ValidationError, "lengths differ"):
            Programme(time=(0.0, 1.0), value=(1.0,))

    def test_time_must_strictly_increase(self):
        for time in ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0, 1.0)):
            with self.subTest(time=time):
                with self.assertRaisesRegex(
                    pyd.ValidationError, "strictly increasing"
                ):
                    Programme(time=time, value=(1.0,) * len(time))

    def test_time_must_be_finite(self):
        for bad_time in (math.nan, math.inf):
            with self.subTest(bad_time=bad_time):
                with self.assertRaises(pyd.ValidationError):
                    Programme(time=(0.0, bad_time), value=(1.0, 2.0))

    def test_non_finite_values_rejected(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(pyd.ValidationError):
                    _programme(value=(1.0, bad_value))

    def test_non_finite_strings_rejected(self):
        for bad_value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(pyd.ValidationError):
                    Programme.model_validate_json(
                        f'{{"time": [0, 1], "value": [1.0, "{bad_value}"]}}'
                    )

    def test_json_round_trip(self):
        programme = _programme(
            value=(1.0, 2.0, 3.0),
            interpolator="Akima1DInterpolator",
            interpolator_kwargs={"method": "makima"},
        )
        restored = Programme.model_validate_json(programme.model_dump_json())
        self.assertEqual(restored.time, programme.time)
        self.assertEqual(restored.interpolator, programme.interpolator)
        self.assertEqual(
            restored.interpolator_kwargs, programme.interpolator_kwargs
        )
        self.assertEqual(restored.value, programme.value)

    def test_unknown_interpolator_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            _programme(interpolator="CubicSpline")

    def test_interpolator_kwargs_accepted(self):
        accepted = {
            "interp1d": {"kind": "linear", "bounds_error": False},
            "Akima1DInterpolator": {"method": "makima", "extrapolate": True},
            "PchipInterpolator": {"extrapolate": False},
            "DerivativeInterpolator": {"left": 0.0, "right": 1.0},
        }
        for interpolator, kwargs in accepted.items():
            with self.subTest(interpolator=interpolator):
                programme = _programme(
                    interpolator=interpolator, interpolator_kwargs=kwargs
                )
                self.assertEqual(programme.interpolator_kwargs, kwargs)

    def test_interpolator_kwargs_rejected(self):
        rejected = {
            "interp1d": {"method": "makima"},
            "Akima1DInterpolator": {"kind": "linear"},
            "PchipInterpolator": {"method": "makima"},
            "DerivativeInterpolator": {"extrapolate": True},
        }
        for interpolator, kwargs in rejected.items():
            with self.subTest(interpolator=interpolator):
                with self.assertRaisesRegex(
                    pyd.ValidationError, "not accepted by"
                ):
                    _programme(
                        interpolator=interpolator, interpolator_kwargs=kwargs
                    )

    def test_frozen(self):
        programme = _programme()
        with self.assertRaises(pyd.ValidationError):
            programme.interpolator = "PchipInterpolator"

    def test_unknown_field_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            Programme(time=(0.0,), value=(1.0,), unit="V")


class TestMagneticProgramme(BLonDTestCase):
    def test_default_unit_is_momentum(self):
        programme = _magnetic(time=(0.0,), value=(1e9,))
        self.assertEqual(programme.in_unit, "momentum")
        self.assertIsNone(programme.bending_radius)

    def test_bending_field_requires_radius(self):
        with self.assertRaisesRegex(pyd.ValidationError, "bending_radius"):
            _magnetic(time=(0.0,), value=(1.0,), in_unit="bending field")

    def test_bending_field_with_radius(self):
        programme = _magnetic(
            time=(0.0,),
            value=(1.0,),
            in_unit="bending field",
            bending_radius=70.0,
        )
        self.assertEqual(programme.bending_radius, 70.0)

    def test_radius_must_be_positive(self):
        for radius in (0.0, -1.0):
            with self.subTest(radius=radius):
                with self.assertRaises(pyd.ValidationError):
                    _magnetic(
                        time=(0.0,),
                        value=(1.0,),
                        in_unit="bending field",
                        bending_radius=radius,
                    )

    def test_unknown_unit_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            _magnetic(time=(0.0,), value=(1.0,), in_unit="speed")

    def test_values_must_be_positive(self):
        for unit in (
            "momentum",
            "total energy",
            "kinetic energy",
            "bending field",
        ):
            for bad_value in (0.0, -1.0):
                with self.subTest(unit=unit, bad_value=bad_value):
                    with self.assertRaisesRegex(
                        pyd.ValidationError, "must be > 0"
                    ):
                        _magnetic(
                            time=(0.0, 1.0),
                            value=(1.0, bad_value),
                            in_unit=unit,
                            bending_radius=70.0
                            if unit == "bending field"
                            else None,
                        )

    def test_non_finite_values_rejected(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(pyd.ValidationError):
                    _magnetic(time=(0.0, 1.0), value=(1.0, bad_value))

    def test_particle_required(self):
        with self.assertRaises(pyd.ValidationError):
            MagneticProgramme(time=(0.0,), value=(1.0,))

    def test_total_energy_must_exceed_mass(self):
        for energy in (938.272e6, 1e6):
            with self.subTest(energy=energy):
                with self.assertRaisesRegex(
                    pyd.ValidationError, "must exceed"
                ):
                    _magnetic(
                        time=(0.0,), value=(energy,), in_unit="total energy"
                    )

    def test_total_energy_above_mass_accepted(self):
        programme = _magnetic(
            time=(0.0,), value=(1e9,), in_unit="total energy"
        )
        self.assertEqual(programme.in_unit, "total energy")

    def test_mass_not_checked_for_other_units(self):
        programme = _magnetic(
            particle=_particle(mass=1e12), time=(0.0,), value=(1e9,)
        )
        self.assertEqual(programme.value, (1e9,))

    def test_to_magnetic_cycle_single_point_is_constant(self):
        programme = _magnetic(time=(0.0,), value=(2e9,))
        cycle = programme.to_magnetic_cycle()
        self.assertIsInstance(cycle, ConstantMagneticCycle)
        self.assertEqual(cycle.value, 2e9)
        self.assertEqual(cycle.in_unit, "momentum")
        self.assertEqual(
            cycle.reference_particle, programme.particle.to_particle_type()
        )

    def test_to_magnetic_cycle_several_points_is_by_time(self):
        programme = _magnetic(
            time=(0.0, 1.0, 2.0),
            value=(1e9, 2e9, 3e9),
            interpolator="Akima1DInterpolator",
            interpolator_kwargs={"method": "makima"},
        )
        cycle = programme.to_magnetic_cycle()
        self.assertIsInstance(cycle, MagneticCycleByTime)
        np.testing.assert_array_equal(cycle.reference_time, (0.0, 1.0, 2.0))
        np.testing.assert_array_equal(cycle.reference_values, (1e9, 2e9, 3e9))
        self.assertIs(cycle.interpolator_type, Akima1DInterpolator)
        self.assertEqual(cycle.interpolator_kwargs, {"method": "makima"})
        self.assertEqual(
            cycle.reference_particle, programme.particle.to_particle_type()
        )

    def test_to_magnetic_cycle_interpolators(self):
        expected = {
            "interp1d": interp1d,
            "Akima1DInterpolator": Akima1DInterpolator,
            "PchipInterpolator": PchipInterpolator,
            "DerivativeInterpolator": DerivativeInterpolator,
        }
        for name, interpolator_type in expected.items():
            with self.subTest(interpolator=name):
                cycle = _magnetic(
                    time=(0.0, 1.0, 2.0),
                    value=(1e9, 2e9, 3e9),
                    interpolator=name,
                ).to_magnetic_cycle()
                self.assertIs(cycle.interpolator_type, interpolator_type)

    def test_to_magnetic_cycle_initial_total_energy(self):
        particle = _particle()
        for time, value in (((0.0,), (2e9,)), ((0.0, 1.0), (2e9, 3e9))):
            with self.subTest(time=time):
                cycle = _magnetic(time=time, value=value).to_magnetic_cycle()
                self.assertAlmostEqual(
                    cycle.get_total_energy_init(),
                    calc_total_energy(mass=particle.mass, momentum=2e9),
                    delta=1e-6,
                )

    def test_to_magnetic_cycle_bending_field(self):
        for time, value in (((0.0,), (1.0,)), ((0.0, 1.0), (1.0, 2.0))):
            with self.subTest(time=time):
                cycle = _magnetic(
                    time=time,
                    value=value,
                    in_unit="bending field",
                    bending_radius=70.0,
                ).to_magnetic_cycle()
                self.assertEqual(cycle.in_unit, "bending field")
                self.assertEqual(cycle.bending_radius, 70.0)

    def test_from_magnetic_cycle_constant(self):
        particle_type = _particle(mass=1.5e9, charge=2).to_particle_type()
        programme = MagneticProgramme.from_magnetic_cycle(
            ConstantMagneticCycle(particle_type, 3e9, in_unit="momentum")
        )
        self.assertEqual(programme.time, (0.0,))
        self.assertEqual(programme.value, (3e9,))
        self.assertEqual(programme.in_unit, "momentum")
        self.assertEqual(programme.particle.mass, 1.5e9)
        self.assertEqual(programme.particle.charge, 2)
        self.assertIsNone(programme.bending_radius)

    def test_from_magnetic_cycle_by_time(self):
        cycle = MagneticCycleByTime(
            _particle().to_particle_type(),
            reference_time=np.array([0.0, 1.0, 2.0]),
            reference_values=np.array([1e9, 2e9, 3e9]),
            interpolator=Akima1DInterpolator,
            method="makima",
        )
        programme = MagneticProgramme.from_magnetic_cycle(cycle)
        self.assertEqual(programme.time, (0.0, 1.0, 2.0))
        self.assertEqual(programme.value, (1e9, 2e9, 3e9))
        self.assertEqual(programme.interpolator, "Akima1DInterpolator")
        self.assertEqual(programme.interpolator_kwargs, {"method": "makima"})

    def test_from_magnetic_cycle_keeps_input_unit(self):
        cycle = MagneticCycleByTime(
            _particle().to_particle_type(),
            reference_time=np.array([0.0, 1.0]),
            reference_values=np.array([1.0, 2.0]),
            in_unit="bending field",
            bending_radius=70.0,
        )
        programme = MagneticProgramme.from_magnetic_cycle(cycle)
        self.assertEqual(programme.in_unit, "bending field")
        self.assertEqual(programme.bending_radius, 70.0)
        self.assertEqual(programme.value, (1.0, 2.0))

    def test_magnetic_cycle_round_trip(self):
        programmes = [
            _magnetic(time=(0.0,), value=(2e9,)),
            _magnetic(
                time=(0.0,),
                value=(1.0,),
                in_unit="bending field",
                bending_radius=70.0,
            ),
            _magnetic(
                time=(0.0, 1.0, 2.0),
                value=(1e9, 2e9, 3e9),
                in_unit="kinetic energy",
                interpolator="PchipInterpolator",
            ),
            _magnetic(
                time=(0.0, 1.0, 2.0),
                value=(1e9, 2e9, 3e9),
                interpolator="Akima1DInterpolator",
                interpolator_kwargs={"method": "makima"},
            ),
            _magnetic(
                time=(0.0, 1.0),
                value=(1e9, 2e9),
                interpolator_kwargs={"kind": "linear"},
            ),
        ]
        for programme in programmes:
            with self.subTest(programme=programme):
                restored = MagneticProgramme.from_magnetic_cycle(
                    programme.to_magnetic_cycle()
                )
                self.assertEqual(restored, programme)

    def test_from_magnetic_cycle_unsupported_cycle(self):
        cycle = MagneticCyclePerTurn(
            _particle().to_particle_type(), 1e9, np.array([1e9, 2e9])
        )
        with self.assertRaises(TypeError):
            MagneticProgramme.from_magnetic_cycle(cycle)

    def test_from_magnetic_cycle_unsupported_interpolator(self):
        cycle = MagneticCycleByTime(
            _particle().to_particle_type(),
            reference_time=np.array([0.0, 1.0, 2.0]),
            reference_values=np.array([1e9, 2e9, 3e9]),
            interpolator=CubicSpline,
        )
        with self.assertRaisesRegex(ValueError, "CubicSpline"):
            MagneticProgramme.from_magnetic_cycle(cycle)

    def test_inherits_programme_checks(self):
        with self.assertRaisesRegex(pyd.ValidationError, "strictly"):
            _magnetic(time=(1.0, 0.0), value=(1.0, 2.0))


class TestParticleDefinition(BLonDTestCase):
    def test_to_particle_type(self):
        particle_type = ParticleDefinition(
            mass=1.5e9, charge=-2, decay_rate=3.0
        ).to_particle_type()
        self.assertIsInstance(particle_type, ParticleType)
        self.assertEqual(particle_type.mass, 1.5e9)
        self.assertEqual(particle_type.charge, -2)
        self.assertEqual(particle_type.user_decay_rate, 3.0)

    def test_from_particle_type(self):
        for particle_type in (proton, electron):
            with self.subTest(particle_type=particle_type):
                particle = ParticleDefinition.from_particle_type(particle_type)
                self.assertEqual(particle.mass, particle_type.mass)
                self.assertEqual(particle.charge, particle_type.charge)
                self.assertIsInstance(particle.charge, int)
                self.assertEqual(
                    particle.decay_rate, particle_type.user_decay_rate
                )

    def test_particle_type_round_trip(self):
        particle_type = ParticleType(mass=2e9, charge=3, user_decay_rate=0.5)
        restored = ParticleDefinition.from_particle_type(
            particle_type
        ).to_particle_type()
        self.assertEqual(restored, particle_type)

    def test_from_particle_type_fractional_charge_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            ParticleDefinition.from_particle_type(
                ParticleType(mass=1.0, charge=1.5)
            )

    def test_defaults(self):
        particle = ParticleDefinition(mass=1.0, charge=1)
        self.assertEqual(particle.decay_rate, 0.0)
        self.assertIsInstance(particle.decay_rate, float)

    def test_negative_charge_accepted(self):
        self.assertEqual(ParticleDefinition(mass=1.0, charge=-1).charge, -1)

    def test_zero_charge_rejected(self):
        for charge in (0, 0.0):
            with self.subTest(charge=charge):
                with self.assertRaisesRegex(pyd.ValidationError, "charge"):
                    ParticleDefinition(mass=1.0, charge=charge)

    def test_integer_float_charge_converted(self):
        particle = ParticleDefinition(mass=1.0, charge=82.0)
        self.assertEqual(particle.charge, 82)
        self.assertIsInstance(particle.charge, int)

    def test_integer_float_charge_converted_from_json(self):
        particle = ParticleDefinition.model_validate_json(
            '{"mass": 1.0, "charge": 2.0}'
        )
        self.assertIsInstance(particle.charge, int)
        self.assertEqual(particle.charge, 2)

    def test_fractional_charge_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            ParticleDefinition(mass=1.0, charge=1.5)

    def test_mass_must_be_positive(self):
        for mass in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(mass=mass):
                with self.assertRaises(pyd.ValidationError):
                    ParticleDefinition(mass=mass, charge=1)

    def test_decay_rate_non_negative(self):
        self.assertEqual(
            ParticleDefinition(mass=1.0, charge=1, decay_rate=0.0).decay_rate,
            0.0,
        )
        self.assertEqual(
            ParticleDefinition(mass=1.0, charge=1, decay_rate=2.5).decay_rate,
            2.5,
        )
        with self.assertRaises(pyd.ValidationError):
            ParticleDefinition(mass=1.0, charge=1, decay_rate=-1.0)


class TestRFHarmonicDefinition(BLonDTestCase):
    def test_valid(self):
        harmonic = _harmonic(harmonic=4620.0, voltage=0.0)
        self.assertEqual(harmonic.harmonic.value, (4620.0,))
        self.assertEqual(harmonic.voltage.value, (0.0,))

    def test_harmonic_must_be_positive(self):
        for bad_harmonic in (0.0, -1.0):
            with self.subTest(harmonic=bad_harmonic):
                with self.assertRaisesRegex(pyd.ValidationError, "harmonic"):
                    _harmonic(harmonic=bad_harmonic)

    def test_voltage_must_be_non_negative(self):
        with self.assertRaisesRegex(pyd.ValidationError, "voltage"):
            _harmonic(voltage=-1.0)

    def test_non_finite_values_rejected(self):
        for bad_value in (math.nan, math.inf, -math.inf):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(pyd.ValidationError):
                    _harmonic(harmonic=bad_value)
                with self.assertRaises(pyd.ValidationError):
                    _harmonic(voltage=bad_value)

    def test_missing_programme_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            RFHarmonicDefinition(
                harmonic=_constant(1.0), voltage=_constant(1.0)
            )


class TestRFStationDefinition(BLonDTestCase):
    def test_defaults(self):
        station = RFStationDefinition(harmonics=(_harmonic(),))
        self.assertEqual(station.main_harmonic_idx, 0)
        self.assertEqual(station.section_index, 0)
        self.assertIsNone(station.name)

    def test_no_harmonics_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            RFStationDefinition(harmonics=())

    def test_main_harmonic_idx_in_range(self):
        station = RFStationDefinition(
            harmonics=(_harmonic(), _harmonic(2.0)), main_harmonic_idx=1
        )
        self.assertEqual(station.main_harmonic_idx, 1)

    def test_main_harmonic_idx_out_of_range(self):
        with self.assertRaisesRegex(pyd.ValidationError, "out of range"):
            RFStationDefinition(
                harmonics=(_harmonic(), _harmonic(2.0)), main_harmonic_idx=2
            )

    def test_negative_indices_rejected(self):
        for field in ("main_harmonic_idx", "section_index"):
            with self.subTest(field=field):
                with self.assertRaises(pyd.ValidationError):
                    RFStationDefinition(
                        harmonics=(_harmonic(),), **{field: -1}
                    )

    def test_integer_indices_required(self):
        with self.assertRaises(pyd.ValidationError):
            RFStationDefinition(harmonics=(_harmonic(),), section_index=1.0)

    def test_to_rf_station_single_constant(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic(harmonic=4620.0, voltage=2e6, phi_rf_design=0.3),
            ),
            section_index=1,
            name="cavity",
        ).to_rf_station()
        self.assertIsInstance(station, SingleHarmonicRFStation)
        self.assertEqual(station.harmonic, 4620.0)
        self.assertEqual(station.voltage, 2e6)
        self.assertEqual(station.phi_rf_design, 0.3)
        self.assertEqual(station.section_index, 1)
        self.assertEqual(station.name, "cavity")
        self.assertEqual(station.schedules, {})

    def test_to_rf_station_single_scheduled(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(
                    voltage=_ramp(
                        (0.0, 1.0),
                        (1e6, 3e6),
                        interpolator="PchipInterpolator",
                    )
                ),
            )
        ).to_rf_station()
        self.assertEqual(set(station.schedules), {"voltage"})
        schedule = station.schedules["voltage"]
        self.assertIsInstance(schedule, ScheduledInterpolation)
        self.assertIs(schedule.interpolator_type, PchipInterpolator)
        self.assertEqual(station.voltage, 1e6)
        station.apply_schedules(turn_i=0, reference_time=0.5)
        self.assertAlmostEqual(station.voltage, 2e6)

    def test_to_rf_station_single_derivative_interpolator(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(
                    voltage=_ramp(
                        (0.0, 1.0, 2.0),
                        (1e6, 2e6, 3e6),
                        interpolator="DerivativeInterpolator",
                    )
                ),
            )
        ).to_rf_station()
        self.assertIs(
            station.schedules["voltage"].interpolator_type,
            DerivativeInterpolator,
        )

    def test_to_rf_station_multi_constant(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic(harmonic=1.0, voltage=1e6, phi_rf_design=0.0),
                _harmonic(harmonic=2.0, voltage=5e5, phi_rf_design=3.14),
            ),
            main_harmonic_idx=1,
        ).to_rf_station()
        self.assertIsInstance(station, MultiHarmonicRFStation)
        self.assertEqual(station.main_harmonic_idx, 1)
        np.testing.assert_array_equal(station.harmonic, [1.0, 2.0])
        np.testing.assert_array_equal(station.voltage, [1e6, 5e5])
        np.testing.assert_array_equal(station.phi_rf_design, [0.0, 3.14])
        self.assertEqual(station.schedules, {})

    def test_to_rf_station_multi_scheduled_shared_axis(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(voltage=_ramp((0.0, 1.0), (1e6, 3e6))),
                _harmonic_of(voltage=_ramp((0.0, 1.0), (4e5, 8e5))),
            )
        ).to_rf_station()
        self.assertEqual(set(station.schedules), {"voltage"})
        schedule = station.schedules["voltage"]
        np.testing.assert_array_equal(schedule.times, [0.0, 1.0])
        np.testing.assert_array_equal(
            schedule.values, [[1e6, 4e5], [3e6, 8e5]]
        )
        np.testing.assert_array_equal(station.voltage, [1e6, 4e5])
        station.apply_schedules(turn_i=0, reference_time=0.5)
        np.testing.assert_allclose(station.voltage, [2e6, 6e5])

    def test_to_rf_station_multi_scheduled_other_attributes(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(phi_rf_design=_ramp((0.0, 1.0), (0.0, 1.0))),
                _harmonic_of(harmonic=_ramp((0.0, 1.0), (2.0, 4.0))),
            )
        ).to_rf_station()
        self.assertEqual(set(station.schedules), {"phi_rf_design", "harmonic"})
        np.testing.assert_array_equal(station.phi_rf_design, [0.0, 0.0])
        np.testing.assert_array_equal(station.harmonic, [1.0, 2.0])

    def test_to_rf_station_multi_common_time_axis(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(voltage=_ramp((0.0, 2.0), (0.0, 2.0))),
                _harmonic_of(voltage=_ramp((0.0, 1.0, 2.0), (0.0, 10.0, 0.0))),
            )
        ).to_rf_station()
        schedule = station.schedules["voltage"]
        np.testing.assert_array_equal(schedule.times, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            schedule.values, [[0.0, 0.0], [1.0, 10.0], [2.0, 0.0]]
        )

    def test_to_rf_station_multi_constant_next_to_ramp(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(voltage=_constant(1e6)),
                _harmonic_of(voltage=_ramp((0.0, 1.0), (0.0, 2.0))),
            )
        ).to_rf_station()
        np.testing.assert_array_equal(
            station.schedules["voltage"].values, [[1e6, 0.0], [1e6, 2.0]]
        )

    def test_to_rf_station_multi_uses_main_harmonic_interpolator(self):
        with self.assertWarnsRegex(UserWarning, "PchipInterpolator"):
            station = RFStationDefinition(
                harmonics=(
                    _harmonic_of(voltage=_ramp((0.0, 1.0), (0.0, 1.0))),
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0),
                            (0.0, 1.0),
                            interpolator="PchipInterpolator",
                        )
                    ),
                ),
                main_harmonic_idx=1,
            ).to_rf_station()
        self.assertIs(
            station.schedules["voltage"].interpolator_type,
            PchipInterpolator,
        )

    def test_to_rf_station_mismatch_warning_names_attribute(self):
        with self.assertWarns(UserWarning) as context:
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(voltage=_ramp((0.0, 1.0), (0.0, 1.0))),
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0),
                            (0.0, 1.0),
                            interpolator="PchipInterpolator",
                        )
                    ),
                )
            ).to_rf_station()
        message = str(context.warning)
        for expected in ("voltage", "main"):
            self.assertIn(expected, message)

    def test_to_rf_station_equal_interpolators_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(voltage=_ramp((0.0, 1.0), (0.0, 1.0))),
                    _harmonic_of(voltage=_ramp((0.0, 1.0), (1.0, 2.0))),
                    _harmonic_of(voltage=_constant(1e6)),
                )
            ).to_rf_station()

    def test_to_rf_station_multi_interpolator_kwargs_of_main(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(
                    voltage=_ramp(
                        (0.0, 1.0, 2.0),
                        (0.0, 1.0, 4.0),
                        interpolator="Akima1DInterpolator",
                        interpolator_kwargs={"method": "makima"},
                    )
                ),
                _harmonic_of(
                    voltage=_ramp(
                        (0.0, 1.0, 2.0),
                        (1.0, 1.0, 1.0),
                        interpolator="Akima1DInterpolator",
                        interpolator_kwargs={"method": "makima"},
                    )
                ),
            )
        ).to_rf_station()
        schedule = station.schedules["voltage"]
        self.assertIs(schedule.interpolator_type, Akima1DInterpolator)
        self.assertEqual(schedule.interpolator_kwargs["method"], "makima")

    def test_to_rf_station_multi_interpolates_along_time_axis(self):
        for name in ("interp1d", "Akima1DInterpolator", "PchipInterpolator"):
            with self.subTest(interpolator=name):
                station = RFStationDefinition(
                    harmonics=(
                        _harmonic_of(
                            voltage=_ramp(
                                (0.0, 1.0, 2.0),
                                (0.0, 1.0, 2.0),
                                interpolator=name,
                            )
                        ),
                        _harmonic_of(
                            voltage=_ramp(
                                (0.0, 1.0, 2.0),
                                (10.0, 10.0, 10.0),
                                interpolator=name,
                            )
                        ),
                    )
                ).to_rf_station()
                station.apply_schedules(turn_i=0, reference_time=1.5)
                np.testing.assert_allclose(station.voltage, [1.5, 10.0])

    def test_to_rf_station_multi_derivative_interpolator_rejected(self):
        with self.assertRaisesRegex(ValueError, "DerivativeInterpolator"):
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0, 2.0),
                            (0.0, 1.0, 2.0),
                            interpolator="DerivativeInterpolator",
                        )
                    ),
                    _harmonic_of(),
                )
            ).to_rf_station()

    def test_from_rf_station_unsupported_type(self):
        with self.assertRaises(TypeError):
            RFStationDefinition.from_rf_station(object())

    def test_from_rf_station_unset_attribute(self):
        with self.assertRaisesRegex(ValueError, "voltage"):
            RFStationDefinition.from_rf_station(
                SingleHarmonicRFStation(phi_rf=0.0, harmonic=1.0)
            )

    def test_from_rf_station_unsupported_schedule(self):
        station = SingleHarmonicRFStation(
            voltage=1e6, phi_rf=0.0, harmonic=1.0
        )
        station.schedule("voltage", np.array([1e6, 2e6]))
        self.assertIsInstance(station.schedules["voltage"], ScheduledArray)
        with self.assertRaisesRegex(TypeError, "voltage"):
            RFStationDefinition.from_rf_station(station)

    def test_from_rf_station_keeps_station_attributes(self):
        station = MultiHarmonicRFStation(
            n_harmonics=2,
            main_harmonic_idx=1,
            voltage=np.array([1e6, 2e6]),
            phi_rf=np.array([0.0, 0.5]),
            harmonic=np.array([1.0, 2.0]),
            section_index=3,
            name="two",
        )
        definition = RFStationDefinition.from_rf_station(station)
        self.assertEqual(definition.main_harmonic_idx, 1)
        self.assertEqual(definition.section_index, 3)
        self.assertEqual(definition.name, "two")
        self.assertEqual(definition.harmonics[1].voltage.value, (2e6,))
        self.assertEqual(definition.harmonics[1].phi_rf_design.value, (0.5,))

    def test_rf_station_round_trip(self):
        definitions = [
            RFStationDefinition(
                harmonics=(_harmonic(4620.0, 2e6, 0.3),),
                section_index=1,
                name="single",
            ),
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0, 2.0),
                            (1e6, 2e6, 3e6),
                            interpolator="PchipInterpolator",
                        )
                    ),
                ),
                name="ramp",
            ),
            RFStationDefinition(
                harmonics=(
                    _harmonic(1.0, 1e6, 0.0),
                    _harmonic(2.0, 5e5, 3.0),
                ),
                main_harmonic_idx=1,
                name="multi",
            ),
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(voltage=_ramp((0.0, 1.0), (1e6, 3e6))),
                    _harmonic_of(
                        voltage=_ramp((0.0, 1.0), (4e5, 8e5)),
                        phi_rf_design=_constant(3.0),
                    ),
                ),
                section_index=2,
                name="shared",
            ),
            RFStationDefinition(
                harmonics=(
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0, 2.0),
                            (0.0, 1.0, 4.0),
                            interpolator="Akima1DInterpolator",
                            interpolator_kwargs={"method": "makima"},
                        )
                    ),
                    _harmonic_of(
                        voltage=_ramp(
                            (0.0, 1.0, 2.0),
                            (2.0, 2.0, 2.0),
                            interpolator="Akima1DInterpolator",
                            interpolator_kwargs={"method": "makima"},
                        )
                    ),
                ),
                name="akima",
            ),
        ]
        for definition in definitions:
            with self.subTest(definition=definition):
                restored = RFStationDefinition.from_rf_station(
                    definition.to_rf_station()
                )
                self.assertEqual(restored, definition)

    def test_from_rf_station_uses_common_time_axis(self):
        station = RFStationDefinition(
            harmonics=(
                _harmonic_of(voltage=_ramp((0.0, 2.0), (0.0, 2.0))),
                _harmonic_of(voltage=_ramp((0.0, 1.0, 2.0), (0.0, 10.0, 0.0))),
            )
        ).to_rf_station()
        restored = RFStationDefinition.from_rf_station(station)
        for harmonic in restored.harmonics:
            self.assertEqual(harmonic.voltage.time, (0.0, 1.0, 2.0))
        self.assertEqual(restored.harmonics[0].voltage.value, (0.0, 1.0, 2.0))
        self.assertEqual(restored.harmonics[1].voltage.value, (0.0, 10.0, 0.0))


class TestDriftDefinition(BLonDTestCase):
    def test_defaults(self):
        drift = DriftDefinition(
            orbit_length=10.0, momentum_compaction_factor=_constant(0.1)
        )
        self.assertIsNone(drift.higher_order_alpha)
        self.assertEqual(drift.section_index, 0)
        self.assertIsNone(drift.name)

    def test_higher_order_alpha(self):
        drift = DriftDefinition(
            orbit_length=10.0,
            momentum_compaction_factor=_constant(0.1),
            higher_order_alpha=(_constant(1e-3), _constant(1e-5)),
        )
        self.assertEqual(len(drift.higher_order_alpha), 2)
        self.assertEqual(drift.higher_order_alpha[1].value, (1e-5,))

    def test_orbit_length_must_be_positive(self):
        for orbit_length in (0.0, -1.0, math.nan):
            with self.subTest(orbit_length=orbit_length):
                with self.assertRaises(pyd.ValidationError):
                    DriftDefinition(
                        orbit_length=orbit_length,
                        momentum_compaction_factor=_constant(0.1),
                    )

    def test_radiation_integrals_default_none(self):
        self.assertIsNone(_drift().radiation_integrals)

    def test_radiation_integrals_stored_as_tuple(self):
        drift = _drift(radiation_integrals=np.array([1.0, 2.0]))
        self.assertEqual(drift.radiation_integrals, (1.0, 2.0))

    def test_radiation_integrals_must_be_finite(self):
        for value in (math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(pyd.ValidationError):
                    _drift(radiation_integrals=(1.0, value))

    def test_to_drift_constant(self):
        drift = _drift(
            orbit_length=50.0, section_index=2, name="arc"
        ).to_drift()
        self.assertIs(type(drift), DriftSimple)
        self.assertEqual(drift.orbit_length, 50.0)
        self.assertEqual(drift.momentum_compaction_factor, 0.1)
        self.assertEqual(drift.section_index, 2)
        self.assertEqual(drift.name, "arc")
        self.assertEqual(drift.schedules, {})
        self.assertIsNone(drift.radiation_integrals)

    def test_to_drift_scheduled_momentum_compaction_factor(self):
        drift = _drift(
            momentum_compaction_factor=_ramp(
                (0.0, 1.0), (0.1, 0.2), interpolator="PchipInterpolator"
            )
        ).to_drift()
        schedule = drift.schedules["momentum_compaction_factor"]
        self.assertIsInstance(schedule, ScheduledInterpolation)
        np.testing.assert_array_equal(schedule.times, (0.0, 1.0))
        np.testing.assert_array_equal(schedule.values, (0.1, 0.2))
        self.assertIs(schedule.interpolator_type, PchipInterpolator)

    def test_to_drift_radiation_integrals(self):
        drift = _drift(radiation_integrals=(1.0, 2.0)).to_drift()
        np.testing.assert_array_equal(drift.radiation_integrals, (1.0, 2.0))

    def test_to_drift_higher_order_constant_is_exact(self):
        drift = _drift(
            higher_order_alpha=(_constant(1e-3), _constant(1e-5))
        ).to_drift()
        self.assertIs(type(drift), DriftExact)
        np.testing.assert_array_equal(drift.higher_order_alpha, (1e-3, 1e-5))
        self.assertNotIn("higher_order_alpha", drift.schedules)

    def test_to_drift_higher_order_scheduled_on_union_of_times(self):
        drift = _drift(
            higher_order_alpha=(
                _ramp((0.0, 2.0), (0.0, 2.0)),
                _ramp((0.0, 1.0, 2.0), (0.0, 10.0, 0.0)),
            )
        ).to_drift()
        schedule = drift.schedules["higher_order_alpha"]
        np.testing.assert_array_equal(schedule.times, (0.0, 1.0, 2.0))
        np.testing.assert_array_equal(
            schedule.values, [[0.0, 0.0], [1.0, 10.0], [2.0, 0.0]]
        )

    def test_to_drift_higher_order_uses_first_order_interpolator(self):
        drift = _drift(
            higher_order_alpha=(
                _ramp(
                    (0.0, 1.0), (0.0, 1.0), interpolator="PchipInterpolator"
                ),
                _ramp((0.0, 1.0), (0.0, 1.0)),
            )
        )
        with self.assertWarns(UserWarning):
            schedule = drift.to_drift().schedules["higher_order_alpha"]
        self.assertIs(schedule.interpolator_type, PchipInterpolator)

    def test_from_drift_simple_round_trip(self):
        original = _drift(
            orbit_length=50.0,
            momentum_compaction_factor=_ramp((0.0, 1.0), (0.1, 0.2)),
            radiation_integrals=(1.0, 2.0),
            section_index=1,
            name="arc",
        )
        self.assertEqual(
            DriftDefinition.from_drift(original.to_drift()), original
        )

    def test_from_drift_exact_round_trip(self):
        original = _drift(
            higher_order_alpha=(
                _ramp((0.0, 1.0), (1e-3, 2e-3)),
                _ramp((0.0, 1.0), (1e-5, 2e-5)),
            ),
            name="arc",
        )
        self.assertEqual(
            DriftDefinition.from_drift(original.to_drift()), original
        )

    def test_from_drift_unset_momentum_compaction_factor(self):
        with self.assertRaisesRegex(ValueError, "momentum_compaction_factor"):
            DriftDefinition.from_drift(DriftSimple(orbit_length=10.0))

    def test_from_drift_other_drift_subclass_rejected(self):
        drift = _DriftLikeLineSegment(
            orbit_length=10.0, momentum_compaction_factor=0.1
        )
        with self.assertRaises(TypeError):
            DriftDefinition.from_drift(drift)

    def test_from_drift_unsupported_type(self):
        with self.assertRaises(TypeError):
            DriftDefinition.from_drift(object())

    def test_from_drift_unsupported_schedule(self):
        drift = DriftSimple(orbit_length=10.0)
        drift.schedule(
            "momentum_compaction_factor",
            ScheduledArray(np.array([0.1, 0.2])),
        )
        with self.assertRaises(TypeError):
            DriftDefinition.from_drift(drift)


class TestCycleDefinition(BLonDTestCase):
    def test_defaults(self):
        cycle = CycleDefinition(
            circumference=1.0,
            magnetic_cycle=_magnetic(time=(0.0,), value=(1.0,)),
        )
        self.assertEqual(cycle.rf_stations, ())
        self.assertEqual(cycle.drifts, ())
        self.assertIsNone(cycle.name)
        self.assertEqual(cycle.schema_version, SCHEMA_VERSION)

    def test_circumference_must_be_positive(self):
        for circumference in (0.0, -1.0, math.inf):
            with self.subTest(circumference=circumference):
                with self.assertRaises(pyd.ValidationError):
                    _cycle(circumference=circumference)

    def test_only_schema_version_one_accepted(self):
        with self.assertRaises(pyd.ValidationError):
            _cycle(schema_version=2)

    def test_particle_belongs_to_magnetic_cycle(self):
        with self.assertRaises(pyd.ValidationError):
            _cycle(particle=_particle())

    def test_frozen(self):
        cycle = _cycle()
        with self.assertRaises(pyd.ValidationError):
            cycle.circumference = 5.0

    def test_unknown_field_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            _cycle(unknown=1)

    def test_json_round_trip(self):
        cycle = _cycle(
            name="test",
            magnetic_cycle=_magnetic(
                time=(0.0, 1.0, 2.0),
                value=(1e9, 2e9, 3e9),
                interpolator="PchipInterpolator",
            ),
        )
        restored = CycleDefinition.model_validate_json(cycle.model_dump_json())
        self.assertEqual(restored.name, "test")
        self.assertEqual(
            restored.magnetic_cycle.particle,
            cycle.magnetic_cycle.particle,
        )
        self.assertEqual(restored.rf_stations, cycle.rf_stations)
        self.assertEqual(restored.drifts, cycle.drifts)
        self.assertEqual(restored.magnetic_cycle.time, (0.0, 1.0, 2.0))
        self.assertEqual(restored.magnetic_cycle.value, (1e9, 2e9, 3e9))

    def test_json_schema_values_are_plain_numbers(self):
        schema = CycleDefinition.model_json_schema()
        self.assertEqual(schema["title"], "CycleDefinition")
        value_schema = schema["$defs"]["MagneticProgramme"]["properties"][
            "value"
        ]
        self.assertEqual(value_schema["items"], {"type": "number"})

    def test_one_rf_station_per_section(self):
        with self.assertRaisesRegex(pyd.ValidationError, "section"):
            _cycle(rf_stations=(_station(), _station()))

    def test_rf_stations_in_different_sections_accepted(self):
        cycle = _cycle(
            rf_stations=(_station(0), _station(1)),
            drifts=(
                _drift(orbit_length=50.0, section_index=0),
                _drift(orbit_length=50.0, section_index=1),
            ),
        )
        self.assertEqual(len(cycle.rf_stations), 2)

    def test_drift_lengths_must_sum_to_circumference(self):
        with self.assertRaisesRegex(pyd.ValidationError, "circumference"):
            _cycle(drifts=(_drift(orbit_length=99.0),))

    def test_circumference_not_checked_without_drifts(self):
        self.assertEqual(_cycle(drifts=()).drifts, ())

    def test_to_ring_orders_by_section_then_type(self):
        ring = _two_section_cycle().to_ring()
        self.assertIsInstance(ring, Ring)
        self.assertEqual(ring.circumference, 100.0)
        self.assertEqual(
            [element.name for element in ring.elements.elements],
            ["rf0", "d0", "rf1", "d1"],
        )

    def test_to_ring_radiation_integrals_sum_of_drifts(self):
        cycle = _cycle(
            drifts=(
                _drift(orbit_length=40.0, radiation_integrals=(1.0, 2.0)),
                _drift(orbit_length=60.0, radiation_integrals=(3.0, 4.0)),
            )
        )
        np.testing.assert_array_equal(
            cycle.to_ring().radiation_integrals, (4.0, 6.0)
        )

    def test_to_ring_radiation_integrals_none_unless_all_drifts_have_them(
        self,
    ):
        cycle = _cycle(
            drifts=(
                _drift(orbit_length=40.0, radiation_integrals=(1.0, 2.0)),
                _drift(orbit_length=60.0),
            )
        )
        self.assertIsNone(cycle.to_ring().radiation_integrals)
        self.assertIsNone(_cycle().to_ring().radiation_integrals)

    def test_from_ring_round_trip(self):
        cycle = _two_section_cycle(name="cycle")
        restored = CycleDefinition.from_ring(
            cycle.to_ring(),
            cycle.magnetic_cycle.to_magnetic_cycle(),
            name="cycle",
        )
        self.assertEqual(restored, cycle)

    def test_from_ring_name_defaults_to_none(self):
        cycle = _two_section_cycle()
        restored = CycleDefinition.from_ring(
            cycle.to_ring(), cycle.magnetic_cycle.to_magnetic_cycle()
        )
        self.assertIsNone(restored.name)

    def test_from_ring_skips_unsupported_elements_with_warning(self):
        cycle = _two_section_cycle()
        ring = cycle.to_ring()
        ring.add_element(_OtherElement(section_index=0, name="other"))
        with self.assertWarnsRegex(UserWarning, "other"):
            restored = CycleDefinition.from_ring(
                ring, cycle.magnetic_cycle.to_magnetic_cycle()
            )
        self.assertEqual(restored, cycle)

    def test_from_ring_skips_other_drift_types_with_warning(self):
        ring = Ring(100.0)
        ring.add_element(
            _DriftLikeLineSegment(
                orbit_length=100.0,
                momentum_compaction_factor=0.1,
                name="segment",
            )
        )
        with self.assertWarnsRegex(UserWarning, "segment"):
            restored = CycleDefinition.from_ring(
                ring, _magnetic(time=(0.0,), value=(1e9,)).to_magnetic_cycle()
            )
        self.assertEqual(restored.drifts, ())
