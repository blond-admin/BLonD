import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator

from blond import (
    DriftSimple,
    MagneticCycleByTime,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    proton,
)


class TestGPUDev(unittest.TestCase):
    def test_interp_nan(self):
        C = 2 * np.pi * 100.0  # CERN PS
        ring = Ring(circumference=C)
        drift = DriftSimple(orbit_length=C)
        drift.momentum_compaction_factor = momentum_compaction_factor(
            transition_gamma=6.1161
        )
        rf = SingleHarmonicRFStation()
        rf.harmonic, rf.voltage, rf.phi_rf_design = 8, 48e3, np.pi
        ring.add_elements((drift, rf), reorder=True)

        t = np.linspace(0.0, 0.5, 501)  # [s]
        p = np.linspace(2.79e9, 20.2e9, 501)  # [eV/c]

        for interp in (None, PchipInterpolator):
            kw = {} if interp is None else {"interpolator": interp}
            cycle = MagneticCycleByTime(
                reference_particle=proton,
                reference_time=t,
                reference_values=p,
                in_unit="momentum",
                **kw,
            )
            Simulation(ring=ring, magnetic_cycle=cycle)
            print(
                f"interpolator={getattr(interp, '__name__', '<default>'):18s} -> n_turns = {cycle.n_turns}"
            )
            self.assertEqual(cycle.n_turns, 236762)


if __name__ == "__main__":
    unittest.main()
