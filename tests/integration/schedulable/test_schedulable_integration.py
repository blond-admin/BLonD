# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Integration tests proving that every ``Schedulable`` user-facing element
actually has its scheduled attribute changed turn-by-turn while a full
``Simulation.run_simulation`` main loop is executing.

Unlike ``tests/unittests/core/test_base.py::TestSchedulable``, which only
calls ``apply_schedules`` directly on a bare helper object, these tests run
a complete ``Simulation`` (``Ring`` + beam + main loop) and record the
scheduled attribute turn-by-turn via a ``run_simulation`` callback. This
proves the scheduling mechanism is actually wired into each element's
``_track`` method and not merely into the ``Schedulable`` base class.
"""

import unittest

import numpy as np
import pytest

from blond import (
    BarrierRF,
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.physics.drifts import DriftExact
from blond.physics.feedbacks.accelerators.lhc import LHCBeamControl
from blond.physics.feedbacks.accelerators.ps import PSBeamControl
from blond.physics.feedbacks.accelerators.psb import PSBBeamControl
from blond.physics.feedbacks.accelerators.sps import SPSBeamControl
from blond.physics.synchrotron_radiation.synchrotron_radiation_master import (
    SynchrotronRadiationMaster,
)

CIRCUMFERENCE = 26658.883
N_TURNS = 5
N_MACROPARTICLES = 100


def _record_each_turn(target, attribute):
    """Build a `run_simulation` callback recording `target.attribute`."""
    recorded = []

    def callback(simulation, beam):
        value = getattr(target, attribute)
        recorded.append(np.array(value, copy=True))

    callback.recorded = recorded
    return callback


class TestSingleHarmonicRFStationScheduling(unittest.TestCase):
    """`RFManipulationBaseClass` schedulable path, via `SingleHarmonicRFStation`."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_voltage_changes_each_turn_during_simulation(self):
        cavity = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
        drift = DriftSimple(
            orbit_length=CIRCUMFERENCE,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=55.759505
            ),
        )
        ring = Ring(CIRCUMFERENCE)
        ring.add_elements([cavity, drift])

        magnetic_cycle = ConstantMagneticCycle(
            value=450e9, reference_particle=proton
        )
        beam = Beam(intensity=1e9, particle_type=proton)

        voltage_schedule = np.linspace(6e6, 8e6, N_TURNS)
        cavity.schedule(attribute="voltage", value=voltage_schedule)

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=0.1e-9, n_macroparticles=N_MACROPARTICLES
            ),
        )

        callback = _record_each_turn(cavity, "voltage")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(callback.recorded, voltage_schedule)
        self.assertNotEqual(callback.recorded[0], callback.recorded[-1])


class TestMultiHarmonicRFStationScheduling(unittest.TestCase):
    """`RFManipulationBaseClass` schedulable path, via `MultiHarmonicRFStation`."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_voltage_changes_each_turn_during_simulation(self):
        cavity = MultiHarmonicRFStation(
            voltage=np.array([6e6]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([35640]),
            n_harmonics=1,
            main_harmonic_idx=0,
        )
        drift = DriftSimple(
            orbit_length=CIRCUMFERENCE,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=55.759505
            ),
        )
        ring = Ring(CIRCUMFERENCE)
        ring.add_elements([cavity, drift])

        magnetic_cycle = ConstantMagneticCycle(
            value=450e9, reference_particle=proton
        )
        beam = Beam(intensity=1e9, particle_type=proton)

        voltage_schedule = np.array(
            [np.linspace(6e6, 8e6, N_TURNS)]
        ).T  # shape (N_TURNS, 1), one value per harmonic per turn
        cavity.schedule(attribute="voltage", value=voltage_schedule)

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=0.1e-9, n_macroparticles=N_MACROPARTICLES
            ),
        )

        callback = _record_each_turn(cavity, "voltage")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(
            np.array(callback.recorded), voltage_schedule
        )
        self.assertFalse(
            np.array_equal(callback.recorded[0], callback.recorded[-1])
        )


class TestBarrierRFScheduling(unittest.TestCase):
    """`RFManipulationBaseClass` schedulable path, via `BarrierRF`."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_peak_voltage_changes_each_turn_during_simulation(self):
        circumference = 2 * np.pi * 100
        momentum = 3.9051e9
        transition_gamma = 6.1

        main_rf = SingleHarmonicRFStation(harmonic=16, phi_rf=0)
        main_rf.voltage = 10e3

        magnetic_cycle = ConstantMagneticCycle(
            value=momentum, reference_particle=proton, in_unit="momentum"
        )
        t_rev = magnetic_cycle.get_t_rev_init(circumference, proton)

        barrier_rf = BarrierRF(t_center=t_rev / 2, t_width=200e-9, n_bins=64)
        peak_voltage_schedule = np.linspace(-5e3, -1e3, N_TURNS)
        barrier_rf.schedule(
            attribute="peak_voltage", value=peak_voltage_schedule
        )

        drift = DriftSimple(
            orbit_length=circumference,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma
            ),
        )

        ring = Ring(circumference)
        ring.add_elements([main_rf, barrier_rf, drift])

        beam = Beam(intensity=1e8, particle_type=proton)

        sim = Simulation(ring, magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=t_rev / 200,
                sigma_dE=1e6,
                n_macroparticles=N_MACROPARTICLES,
            ),
        )

        callback = _record_each_turn(barrier_rf, "peak_voltage")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(callback.recorded, peak_voltage_schedule)
        self.assertNotEqual(callback.recorded[0], callback.recorded[-1])


class TestDriftSimpleScheduling(unittest.TestCase):
    """`DriftSimple.momentum_compaction_factor` scheduling."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_momentum_compaction_factor_changes_each_turn(self):
        cavity = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
        drift = DriftSimple(orbit_length=CIRCUMFERENCE)
        ring = Ring(CIRCUMFERENCE)
        ring.add_elements([cavity, drift])

        magnetic_cycle = ConstantMagneticCycle(
            value=450e9, reference_particle=proton
        )
        beam = Beam(intensity=1e9, particle_type=proton)

        alpha_schedule = np.linspace(
            momentum_compaction_factor(transition_gamma=55.759505),
            momentum_compaction_factor(transition_gamma=60.0),
            N_TURNS,
        )
        drift.schedule(
            attribute="momentum_compaction_factor", value=alpha_schedule
        )

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=0.1e-9, n_macroparticles=N_MACROPARTICLES
            ),
        )

        callback = _record_each_turn(drift, "momentum_compaction_factor")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(callback.recorded, alpha_schedule)
        self.assertNotEqual(callback.recorded[0], callback.recorded[-1])


class TestDriftExactScheduling(unittest.TestCase):
    """`DriftExact.higher_order_alpha` scheduling."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_higher_order_alpha_changes_each_turn(self):
        cavity = SingleHarmonicRFStation(harmonic=35640, voltage=6e6, phi_rf=0)
        drift = DriftExact(
            orbit_length=CIRCUMFERENCE,
            momentum_compaction_factor=momentum_compaction_factor(
                transition_gamma=55.759505
            ),
            higher_order_alpha=np.array([1.0]),
        )
        ring = Ring(CIRCUMFERENCE)
        ring.add_elements([cavity, drift])

        magnetic_cycle = ConstantMagneticCycle(
            value=450e9, reference_particle=proton
        )
        beam = Beam(intensity=1e9, particle_type=proton)

        higher_order_alpha_schedule = np.array(
            [np.linspace(1.0, 2.0, N_TURNS)]
        ).T  # shape (N_TURNS, 1)
        drift.schedule(
            attribute="higher_order_alpha",
            value=higher_order_alpha_schedule,
        )

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=0.1e-9, n_macroparticles=N_MACROPARTICLES
            ),
        )

        callback = _record_each_turn(drift, "higher_order_alpha")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(
            np.array(callback.recorded), higher_order_alpha_schedule
        )
        self.assertFalse(
            np.array_equal(callback.recorded[0], callback.recorded[-1])
        )


class TestSynchrotronRadiationMasterScheduling(unittest.TestCase):
    """`SynchrotronRadiationMaster.radiation_integrals` scheduling.

    Note
    ----
    KNOWN BUG (xfail, not fixed by this test module): `.schedule()` is
    overridden by `SynchrotronRadiationMaster` to propagate the given
    schedule to every generated `_SynchrotronRadiationTracker` child via
    ``SRClass_child.schedule(attribute=attribute, value=...)``. That
    forwards the SAME attribute name given by the caller (documented as
    ``"radiation_integrals"``), but each child only registers
    ``"share_of_radiation_integrals"`` as schedulable
    (``synchrotron_radiation_master.py``, ``_insert_radiation_trackers``
    / child ``_register_schedulable_variables`` call). So calling
    ``SynchrotronRadiationMaster().schedule("radiation_integrals", ...)``
    -- the class's own documented usage -- crashes immediately with
    ``AssertionError: Attribute radiation_integrals doesnt exist`` raised
    from inside the child's ``.schedule()``
    (``blond/core/base.py``, ``Schedulable.schedule``).
    """

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "SynchrotronRadiationMaster.schedule() propagates the caller's "
            "attribute name unchanged to each generated "
            "_SynchrotronRadiationTracker child, but the child only "
            "registers 'share_of_radiation_integrals' as schedulable, not "
            "'radiation_integrals'. Raises AssertionError immediately. "
            "See synchrotron_radiation_master.py, SynchrotronRadiationMaster"
            ".schedule()."
        ),
    )
    def test_radiation_integrals_changes_each_turn_during_simulation(self):
        radiation_integrals = np.array(
            [
                0.646747216157,
                0.0005936549319,
                5.6814536525e-08,
                5.92870407301e-09,
                1.71368060083e-11,
            ]
        )
        circumference = 90.65874532e3

        cavity = SingleHarmonicRFStation(
            harmonic=242400, voltage=50.1e6, phi_rf=0
        )
        drift = DriftSimple(
            orbit_length=circumference,
            momentum_compaction_factor=radiation_integrals[0] / circumference,
        )
        ring = Ring(circumference, radiation_integrals=radiation_integrals)
        ring.add_elements([cavity, drift])

        srm = SynchrotronRadiationMaster()
        srm.prepare_ring_for_synchrotron_radiation_tracking(ring=ring)
        (radiation_tracker,) = srm.generated_children

        radiation_integrals_schedule = np.stack(
            [
                radiation_integrals * (1.0 + 0.1 * turn)
                for turn in range(N_TURNS)
            ]
        )
        # Intended usage per the class's own docstring/registered
        # schedulable variable -- currently raises, see class docstring.
        srm.schedule(
            attribute="radiation_integrals",
            value=radiation_integrals_schedule,
        )

        magnetic_cycle = ConstantMagneticCycle(
            value=20e9,
            reference_particle=proton,
            in_unit="total energy",
        )
        beam = Beam(intensity=2.725e10, particle_type=proton)

        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
        sim.prepare_beam(
            beam=beam,
            preparation_routine=BiGaussian(
                sigma_dt=4e-3 / 3e8, n_macroparticles=N_MACROPARTICLES
            ),
        )

        callback = _record_each_turn(
            radiation_tracker, "share_of_radiation_integrals"
        )
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        expected = (
            radiation_integrals_schedule
            * radiation_tracker.share_of_radiation_integrals
            / radiation_integrals
        )
        np.testing.assert_allclose(np.array(callback.recorded), expected)
        self.assertFalse(
            np.array_equal(callback.recorded[0], callback.recorded[-1])
        )


def _build_beam_feedback_scenario(beam_control_cls, beam_control_kwargs):
    """Assemble a minimal single-bunch ring with a beam-feedback loop."""
    circumference = 26658.8832
    momentum = 450e9
    intensity = 1.6e11
    voltage = 5e6
    harmonic = 35640
    gamma_t = 53.8
    alpha = 1 / gamma_t / gamma_t

    energy = np.sqrt(momentum**2 + proton.mass**2)
    rel_gamma = energy / proton.mass
    rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

    lattice = DriftSimple(
        orbit_length=circumference, momentum_compaction_factor=alpha
    )
    cavity = MultiHarmonicRFStation(
        voltage=np.array([voltage]),
        phi_rf=np.array([0.0]),
        harmonic=np.array([harmonic]),
        n_harmonics=1,
        main_harmonic_idx=0,
    )
    f_rf = cavity.calc_main_harmonic_omega_rf_design(
        rel_beta, lattice.orbit_length
    ) / (2 * np.pi)
    t_rf = 1 / f_rf

    profile = StaticProfile(
        cut_left=-1.5 * t_rf,
        cut_right=2.5 * t_rf,
        n_bins=4 * 2**6,
    )

    beam_control = beam_control_cls(profile=profile, **beam_control_kwargs)
    cavity.attach_beam_feedback(beam_control)

    ring = Ring(circumference)
    ring.add_elements([profile, cavity, beam_control, lattice])

    cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")
    sim = Simulation(ring, cycle)

    beam = Beam(intensity, proton)
    sim.prepare_beam(
        beam,
        BiGaussian(
            n_macroparticles=N_MACROPARTICLES,
            sigma_dt=1.2e-9 / 4,
            seed=1234,
        ),
    )

    return sim, beam, cavity, beam_control


class TestBeamFeedbackScheduling(unittest.TestCase):
    """`BeamFeedbackBase` schedulable path, for every accelerator flavour."""

    def setUp(self):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("numba")

    def _assert_pl_gain_changes_each_turn(
        self, beam_control_cls, beam_control_kwargs
    ):
        sim, beam, _cavity, beam_control = _build_beam_feedback_scenario(
            beam_control_cls, beam_control_kwargs
        )

        pl_gain_schedule = np.linspace(1e3, 2e3, N_TURNS)
        beam_control.schedule(attribute="pl_gain", value=pl_gain_schedule)

        callback = _record_each_turn(beam_control, "pl_gain")
        sim.run_simulation(
            beams=(beam,),
            n_turns=N_TURNS,
            callbacks=callback,
            show_progressbar=False,
            verbose=False,
        )

        np.testing.assert_allclose(callback.recorded, pl_gain_schedule)
        self.assertNotEqual(callback.recorded[0], callback.recorded[-1])

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_lhc_pl_gain_changes_each_turn(self):
        self._assert_pl_gain_changes_each_turn(LHCBeamControl, {})

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_psb_pl_gain_changes_each_turn(self):
        self._assert_pl_gain_changes_each_turn(PSBBeamControl, {})

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_ps_pl_gain_changes_each_turn(self):
        self._assert_pl_gain_changes_each_turn(PSBeamControl, {})

    @pytest.mark.backend_mutation
    @pytest.mark.integration
    def test_sps_pl_gain_changes_each_turn(self):
        self._assert_pl_gain_changes_each_turn(
            SPSBeamControl,
            dict(
                k_phi_n=0.0,
                k_phi_nm1=0.0,
                k_eps_n=0.0,
                k_z_n=0.0,
                k_a_n=0.0,
                k_b_n=0.0,
                phi_sync=0.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
