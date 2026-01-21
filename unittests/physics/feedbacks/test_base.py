import unittest
from unittest.mock import Mock

import numpy as np

from blond import (
    ConstantMagneticCycle,
    MultiHarmonicRFStation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
)
from blond.core.beam.base import BeamBaseClass
from blond.physics.feedbacks.base import GlobalFeedback, LocalFeedback


class TestLocalFeedbackBase(unittest.TestCase):
    def test_error_throwing_on_non_conformant_parent_cavity(self):
        class LocFdbkHelper(LocalFeedback):
            def track(self, beam: BeamBaseClass) -> None:
                pass

            def on_run_simulation(
                self,
                simulation: Simulation,
                beam: BeamBaseClass,
                n_turns: int,
                **kwargs,
            ) -> None:
                pass

            def on_init_simulation(self, simulation: Simulation) -> None:
                pass

        prof = Mock(StaticProfile)
        fdbk = LocFdbkHelper(profile=prof)
        with self.assertRaisesRegex(ValueError, "Local feedbacks can only be"):
            fdbk.set_parent_rf_station(prof)


class TestGlobalFeedbackBase(unittest.TestCase):
    def test_cavities_in_list(self):
        class GlobalFdbkHelper(GlobalFeedback):
            def track(self, beam: BeamBaseClass) -> None:
                pass

            def on_run_simulation(
                self,
                simulation: Simulation,
                beam: BeamBaseClass,
                n_turns: int,
                **kwargs,
            ) -> None:
                pass

            def on_init_simulation(self, simulation: Simulation) -> None:
                super().on_init_simulation(simulation=simulation)

        prof = Mock(StaticProfile)
        fdbk = GlobalFdbkHelper(profile=prof)
        ring = Ring(circumference=456, check_section_indices=False)
        shc = SingleHarmonicRFStation(
            section_index=0, voltage=1, harmonic=1, phi_rf=1
        )
        ring.add_element(shc)
        mhc = MultiHarmonicRFStation(
            n_harmonics=1,
            main_harmonic_idx=0,
            section_index=1,
            voltage=np.array([1]),
            harmonic=np.array([1]),
            phi_rf=np.array([1]),
        )
        ring.add_element(mhc)
        sim = Simulation(ring=ring, magnetic_cycle=Mock(ConstantMagneticCycle))

        fdbk.on_init_simulation(simulation=sim)
        assert mhc in fdbk.cavities
        assert shc in fdbk.cavities
