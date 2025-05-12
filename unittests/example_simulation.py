import numpy as np

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker


class ExampleSimulation:
    def __init__(self):
        # Define general parameters
        ring = Ring(
            26658.883,
            1.0 / 55.759505**2,
            np.linspace(450e9, 460.005e9, 2000 + 1),
            Proton(),
            2000,
        )
        self.rf_station = RFStation(ring, [35640], [6e6], [0])

        dt = np.random.randn(1001)
        dt /= dt.max()
        dt *= 0.5
        dt += 0.5
        dt *= (self.rf_station.t_rev[0] / self.rf_station.harmonic[0, 0])
        dE = np.random.randn(1001)
        dE *= 1e9

        # Define beam and distribution
        self.beam = Beam(ring, 1001, 1e9, dt=dt,dE=dE)

        # Define RF station parameters and corresponding tracker
        self.ring_and_rf_tracker = RingAndRFTracker(self.rf_station, self.beam)

        # Need slices for the Gaussian fit
        self.profile = Profile(
            self.beam,
            CutOptions(n_slices=100, cut_left=0, cut_right=self.rf_station.t_rev[0]/ self.rf_station.harmonic[0, 0]),
        )
