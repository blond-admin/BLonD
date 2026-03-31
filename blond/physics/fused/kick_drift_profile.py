import sys
import time
from copy import copy, deepcopy
from typing import Any

import numpy as np

from blond import (
    SingleHarmonicRFStation,
    DriftSimple,
    StaticProfile,
    EmptyBeam,
    proton,
    backend,
    Simulation,
    Ring,
    MagneticCyclePerTurn,
    copy_to_cpu,
)
from blond.core.beam.base import BeamBaseClass
from blond.core.beam.beams import ProbeBeam
from core.reference_clock.reference_clock import ReferenceCoordinates


class KickDriftProfile(SingleHarmonicRFStation, DriftSimple, StaticProfile):
    def __init__(
        self,
        voltage: float = None,
        phi_rf: float = None,
        harmonic: float = None,
        orbit_length: float = None,
        momentum_compaction_factor: float = None,
        cut_left: float = None,
        cut_right: float = None,
        n_bins: int = None,
    ):
        super().__init__(
            voltage=voltage,
            phi_rf=phi_rf,
            harmonic=harmonic,
            orbit_length=orbit_length,
            momentum_compaction_factor=momentum_compaction_factor,
            cut_left=cut_left,
            cut_right=cut_right,
            n_bins=n_bins,
        )

    def _track(self, beam: BeamBaseClass) -> None:
        hack = EmptyBeam(
            particle_type=beam.particle_type,
            reference_time=beam.reference.time,
            reference_total_energy=beam.reference.total_energy,
            intensity=beam.intensity,
        )

        SingleHarmonicRFStation._track(self, beam=hack)

        kwargs = dict(
            dt=beam.write_partial_dt(),
            dE=beam.write_partial_dE(),
            voltage=copy(self.voltage),
            phi_rf=copy(self.phi_rf),
            omega_rf=copy(self.omega_rf),
            charge=beam.particle_type.charge,
            acceleration_kick=-copy(self._last_reference_energy_change),
            # Mind the
            # minus!
        )

        DriftSimple._track(self, beam=hack)

        kwargs.update(
            dict(
                # dt=beam.write_partial_dt(),
                # dE=beam.read_partial_dE(),
                T=self._last_dt,
                eta_0=self._last_eta_0,
                beta=(hack.reference.beta),
                energy=(hack.reference.total_energy),
            )
        )

        #StaticProfile._track(self, beam=hack)

        #kwargs.update(
        #    dict(
        #        array_write=self._hist_y,
        #        start=self.cut_left,
        #        stop=self.cut_right,
        #    )
        # )

        backend.specials.fused_kick_drift_profile(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, float):
                kwargs[key] = val

        #self.hist_y_to_density_factor = 1.0 / beam.common_array_size

        #self.invalidate_cache()

        beam.reference.time = hack.reference.time
        beam.reference.total_energy = hack.reference.total_energy

    def on_init_simulation(self, simulation: Simulation) -> None:
        SingleHarmonicRFStation.on_init_simulation(self, simulation=simulation)
        DriftSimple.on_init_simulation(self, simulation=simulation)
        StaticProfile.on_init_simulation(self, simulation=simulation)

    def on_run_simulation(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_turns: int,
        **kwargs: dict[str, Any],
    ) -> None:
        SingleHarmonicRFStation.on_run_simulation(
            self, simulation=simulation, beam=beam, n_turns=n_turns
        )
        DriftSimple.on_run_simulation(
            self, simulation=simulation, beam=beam, n_turns=n_turns
        )
        StaticProfile.on_run_simulation(
            self, simulation=simulation, beam=beam, n_turns=n_turns
        )

    def track_reference(
        self,
        reference: ReferenceCoordinates,
        is_counter_rotating: bool = False,
    ) -> float:
        return
        SingleHarmonicRFStation.track_reference(
            self, reference=reference, is_counter_rotating=is_counter_rotating
        )
        DriftSimple.track_reference(
            self, reference=reference, is_counter_rotating=is_counter_rotating
        )


class TestKickDriftProfile(object):
    def __init__(self):
        self.obj = KickDriftProfile(
            voltage=1e6,
            phi_rf=1,
            harmonic=3,
            orbit_length=4,
            momentum_compaction_factor=2,
            cut_left=0,
            cut_right=2,
            n_bins=64,
        )
        self.kick = SingleHarmonicRFStation(
            voltage=1e6,
            phi_rf=1,
            harmonic=3,
        )
        self.drift = DriftSimple(
            orbit_length=4,
            momentum_compaction_factor=2,
        )
        self.profile = StaticProfile(
            cut_left=0,
            cut_right=2,
            n_bins=64,
        )

        for obj in (self.obj, self.kick):
            obj.schedule("phi_rf_design", np.array([1, 2]))

        for obj in (self.obj, self.drift):
            obj.schedule("momentum_compaction_factor", np.array([2, 4]))

    def _setup_Sim(self):
        self._setupA()
        self._setupB()

    def _setupA(self):
        print(backend.specials_mode)
        ring = Ring(circumference=123, check_section_indices=False)
        ring.add_element(self.obj)
        cycle = MagneticCyclePerTurn.init_from_linspace(
            reference_particle=proton,
            values=np.linspace(1e12, 1.1e12, num=int(1e3)),
        )
        sim = Simulation(ring=ring, magnetic_cycle=cycle)

    def _setupB(self):
        print(backend.specials_mode)
        ring = Ring(circumference=123, check_section_indices=False)
        ring.add_elements((self.kick, self.drift, self.profile))
        cycle = MagneticCyclePerTurn.init_from_linspace(
            reference_particle=proton,
            values=np.linspace(1e12, 1.1e12, num=int(1e3)),
        )
        sim = Simulation(ring=ring, magnetic_cycle=cycle)

    def test___init__(self):
        print(type(self.obj).__mro__)

    def test_track(self):
        self._setup_Sim()
        beam = ProbeBeam(
            dt=np.linspace(0, 10, 10),
            dE=np.linspace(0, 10, 10),
            particle_type=proton,
            reference_total_energy=1e12,
            reference_time=0.0,
        )
        self.obj.track(beam)

    def test_track_correct_kick_only(self):
        self._setup_Sim()
        actual = ProbeBeam(
            dt=np.linspace(0, 10, 10),
            dE=np.linspace(-10, 10, 10),
            particle_type=proton,
            reference_total_energy=1e12,
            reference_time=0.0,
        )
        desired = deepcopy(actual)

        self.obj.track(actual)
        self.kick.track(desired)
        np.testing.assert_almost_equal(
            actual.dE.copy_as_numpy(),
            desired.dE.copy_as_numpy(),
        )
        np.testing.assert_almost_equal(
            actual.reference.time,
            desired.reference.time,
        )

        np.testing.assert_almost_equal(
            actual.reference.total_energy,
            desired.reference.total_energy,
        )

    def test_track_correct_kickdrift_only(self):
        self._setup_Sim()
        actual = ProbeBeam(
            dt=np.linspace(0, 10, 10),
            dE=np.linspace(-10, 10, 10),
            particle_type=proton,
            reference_total_energy=1e12,
            reference_time=0.0,
        )
        desired = deepcopy(actual)
        self.obj.track(actual)
        self.kick.track(desired)
        self.drift.track(desired)
        np.testing.assert_almost_equal(
            actual.dE.copy_as_numpy(),
            desired.dE.copy_as_numpy(),
        )
        np.testing.assert_almost_equal(
            actual.reference.time,
            desired.reference.time,
        )

        np.testing.assert_almost_equal(
            actual.reference.total_energy,
            desired.reference.total_energy,
        )

        np.testing.assert_almost_equal(
            actual.dt.copy_as_numpy(),
            desired.dt.copy_as_numpy(),
        )

    def test_track_correct_kickdriftprofile(self):
        self._setup_Sim()
        actual = ProbeBeam(
            dt=np.linspace(0, 10, 10),
            dE=np.linspace(-10, 10, 10),
            particle_type=proton,
            reference_total_energy=1e12,
            reference_time=0.0,
        )
        desired = deepcopy(actual)
        for turn in range(2):
            self.obj._simulation.turn_i.value = turn
            self.obj.value = turn
            self.kick._turn_i.value = turn
            self.drift._simulation.turn_i.value = turn

            self.obj.track(actual)
            self.kick.track(desired)
            self.drift.track(desired)
            self.profile.track(desired)
            np.testing.assert_almost_equal(
                actual.dE.copy_as_numpy(),
                desired.dE.copy_as_numpy(),
            )
            np.testing.assert_almost_equal(
                actual.reference.time,
                desired.reference.time,
            )

            np.testing.assert_almost_equal(
                actual.reference.total_energy,
                desired.reference.total_energy,
            )

            np.testing.assert_almost_equal(
                actual.dt.copy_as_numpy(),
                desired.dt.copy_as_numpy(),
            )
            np.testing.assert_almost_equal(
                copy_to_cpu(self.obj.hist_y),
                copy_to_cpu(self.profile.hist_y),
            )
            print(f"Turn {turn} sucessful")


if __name__ == "__main__":
    backend.set_specials("python")
    tests = TestKickDriftProfile()
    tests.test___init__()
    tests.test_track()
    tests.test_track_correct_kick_only()
    tests.test_track_correct_kickdrift_only()
    tests.test_track_correct_kickdriftprofile()
