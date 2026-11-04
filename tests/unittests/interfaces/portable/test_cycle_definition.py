# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Unit tests for the portable cycle definition models."""

import math

import numpy as np
import pytest

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


def _constant(value):
    return Programme(time=(0.0,), value=(value,))


def _harmonic(harmonic=1.0, voltage=1e6, phi_rf_design=0.0):
    return RFHarmonicDefinition(
        harmonic=_constant(harmonic),
        voltage=_constant(voltage),
        phi_rf_design=_constant(phi_rf_design),
    )


def _cycle(**kwargs):
    arguments = {
        "particle": ParticleDefinition(mass=938.272e6, charge=1),
        "circumference": 100.0,
        "magnetic_cycle": MagneticProgramme(
            time=(0.0, 1.0), value=(1e9, 2e9)
        ),
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
        programme = MagneticProgramme(time=(0.0,), value=(1e9,))
        self.assertEqual(programme.in_unit, "momentum")
        self.assertIsNone(programme.bending_radius)

    def test_bending_field_requires_radius(self):
        with self.assertRaisesRegex(pyd.ValidationError, "bending_radius"):
            MagneticProgramme(
                time=(0.0,), value=(1.0,), in_unit="bending field"
            )

    def test_bending_field_with_radius(self):
        programme = MagneticProgramme(
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
                    MagneticProgramme(
                        time=(0.0,),
                        value=(1.0,),
                        in_unit="bending field",
                        bending_radius=radius,
                    )

    def test_unknown_unit_rejected(self):
        with self.assertRaises(pyd.ValidationError):
            MagneticProgramme(time=(0.0,), value=(1.0,), in_unit="speed")

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
                        MagneticProgramme(
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
                    MagneticProgramme(time=(0.0, 1.0), value=(1.0, bad_value))

    def test_inherits_programme_checks(self):
        with self.assertRaisesRegex(pyd.ValidationError, "strictly"):
            MagneticProgramme(time=(1.0, 0.0), value=(1.0, 2.0))


class TestParticleDefinition(BLonDTestCase):
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
                    RFStationDefinition(harmonics=(_harmonic(),), **{field: -1})

    def test_integer_indices_required(self):
        with self.assertRaises(pyd.ValidationError):
            RFStationDefinition(harmonics=(_harmonic(),), section_index=1.0)


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


class TestCycleDefinition(BLonDTestCase):
    def test_defaults(self):
        cycle = CycleDefinition(
            particle=ParticleDefinition(mass=1.0, charge=1),
            circumference=1.0,
            magnetic_cycle=MagneticProgramme(time=(0.0,), value=(1.0,)),
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

    def test_total_energy_must_exceed_mass(self):
        particle = ParticleDefinition(mass=938.272e6, charge=1)
        for energy in (938.272e6, 1e6):
            with self.subTest(energy=energy):
                with self.assertRaisesRegex(
                    pyd.ValidationError, "must exceed"
                ):
                    _cycle(
                        particle=particle,
                        magnetic_cycle=MagneticProgramme(
                            time=(0.0,), value=(energy,), in_unit="total energy"
                        ),
                    )

    def test_total_energy_above_mass_accepted(self):
        cycle = _cycle(
            magnetic_cycle=MagneticProgramme(
                time=(0.0,), value=(1e9,), in_unit="total energy"
            )
        )
        self.assertEqual(cycle.magnetic_cycle.in_unit, "total energy")

    def test_mass_not_checked_for_other_units(self):
        cycle = _cycle(
            particle=ParticleDefinition(mass=1e12, charge=1),
            magnetic_cycle=MagneticProgramme(
                time=(0.0,), value=(1e9,), in_unit="momentum"
            ),
        )
        self.assertEqual(cycle.magnetic_cycle.value, (1e9,))

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
            magnetic_cycle=MagneticProgramme(
                time=(0.0, 1.0, 2.0),
                value=(1e9, 2e9, 3e9),
                interpolator="PchipInterpolator",
            ),
        )
        restored = CycleDefinition.model_validate_json(cycle.model_dump_json())
        self.assertEqual(restored.name, "test")
        self.assertEqual(restored.particle, cycle.particle)
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
