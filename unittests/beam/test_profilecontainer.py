import unittest

import numpy as np
from scipy.constants import c, e, m_p

from blond.beam.beam import Beam, Proton
from blond.beam.profile import CutOptions, Profile
from blond.beam.profile import FitOptions
from blond.beam.profilecontainer import (
    Lockable,
    ProfileContainer,
    InducedVoltageContainer,
    TotalInducedVoltageNew,
)
from blond.impedances.impedance import InducedVoltageTime
from blond.impedances.impedance import (
    InductiveImpedance,
    TotalInducedVoltage,
)
from blond.impedances.impedance_sources import Resonators
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker


class TestLockable(unittest.TestCase):
    def setUp(self):
        self.obj = Lockable()

    def test_lock(self):
        self.obj.lock()

    def test_unlock(self):
        self.obj.unlock()

    def test_is_locked(self):
        self.obj.lock()

        assert self.obj.is_locked is True
        self.obj.unlock()

        assert self.obj.is_locked is True


class TestProfileContainer(unittest.TestCase):
    def setUp(self):
        self.ring = Ring(
            ring_length=125,
            alpha_0=0.001,
            synchronous_data=1e9,
            particle=Proton(),
            n_turns=1,
        )

        # RF object initialization
        self.rf_params = RFStation(
            ring=self.ring, harmonic=[1], voltage=[7e6], phi_rf_d=[0.0], n_rf=1
        )

        # Beam parameters
        my_beam = Beam(self.ring, n_macroparticles=100000, intensity=1e10)

        # First profile object initialization and tracking
        self.profile1 = Profile(
            my_beam,
            cut_options=CutOptions(
                cut_left=0,
                cut_right=10,
                n_slices=100,
            ),
        )

        self.profile_container = ProfileContainer()
        self.profile_container.add_profile(self.profile1)

    def test_bin_width(self):
        assert self.profile_container.bin_width == 0.1

    def test_n_profiles(self):
        assert self.profile_container.n_profiles == 1

    def test_number_of_bins(self):
        assert self.profile_container.number_of_bins == 100

    def test_add_profile(self):
        profile2 = Profile(
            self.profile1.beam,
            cut_options=CutOptions(
                cut_left=10,
                cut_right=20,
                n_slices=100,
            ),
        )
        self.profile_container.add_profile(profile2)

    def test_add_profile_overlap(self):
        self.assertRaises(
            ValueError, lambda: self.profile_container.add_profile(self.profile1)
        )

    def test__update_memory(self):
        self.profile_container._update_memory()

    def test_track(self):
        self.profile_container.track()

    def test___len__(self):
        assert len(self.profile_container) == 1

    def test___iter__(self):
        for profile1 in self.profile_container:
            for profile2 in self.profile_container:
                pass


class TestInducedVoltageContainer(unittest.TestCase):
    def setUp(self):
        ring = Ring(6911.56, 1 / (1 / np.sqrt(0.00192)) ** 2, 25.92e9, Proton(), 1000)
        rf_station = RFStation(
            ring,
            [4620],
            [0.9e6],
            [0.0],
            1,
        )
        beam = Beam(ring, 5 * 1e6, 1e10)
        cut_options = CutOptions(
            cut_left=0,
            cut_right=2 * np.pi,
            n_slices=10,
            rf_station=rf_station,
            cuts_unit="rad",
        )
        slice_beam = Profile(beam, cut_options, FitOptions(fit_option="gaussian"))
        resonator = Resonators(np.ones(10), np.ones(10), np.ones(10))
        ind_volt = InducedVoltageTime(beam, slice_beam, [resonator])

        self.induced_voltage_container = InducedVoltageContainer()
        self.induced_voltage_container.add_induced_voltage(ind_volt)

    def test_n_objects(self):
        assert self.induced_voltage_container.n_objects == 1

    # def test_add_induced_voltage(self):
    # already in setUp
    # self.obj.add_induced_voltage

    def test___len__(self):
        assert len(self.induced_voltage_container) == 1

    def test___iter__(self):
        for indced_voltage in self.induced_voltage_container:
            for indced_voltage in self.induced_voltage_container:
                pass


class TestTotalInducedVoltageNew(unittest.TestCase):
    def setUp(self):
        # SIMULATION PARAMETERS -------------------------------------------------------

        # Beam parameters
        sigma_dt = 180e-9 / 4  # [s]
        kin_beam_energy = 1.4e9  # [eV]

        E_0 = m_p * c**2 / e  # [eV]
        tot_beam_energy = E_0 + kin_beam_energy  # [eV]

        ring = Ring(
            2 * np.pi * 25,
            1 / 4.4**2,
            np.sqrt(tot_beam_energy**2 - E_0**2),
            Proton(),
            2,
        )

        rf_station = RFStation(ring, [1], [8e3], [np.pi], 1)

        dt = np.random.randn(1001) * sigma_dt
        dE = np.random.randn(len(dt)) / sigma_dt
        beam = Beam(ring, len(dt), 1e11, dt=dt, dE=dE)

        # DEFINE BEAM------------------------------------------------------------------
        # ring_RF_section = RingAndRFTracker(rf_station, beam)
        # bigaussian(ring, rf_station, my_beam, sigma_dt, seed=1)

        # DEFINE SLICES----------------------------------------------------------------
        profile1 = Profile(
            beam,
            CutOptions(cut_left=-sigma_dt, cut_right=sigma_dt, n_slices=64),
        )

        steps = InductiveImpedance(
            beam,
            profile1,
            34.6669349520904 / 10e9 * ring.f_rev,
            rf_station,
            deriv_mode="diff",
        )
        # direct space charge
        dir_space_charge = InductiveImpedance(
            beam,
            profile1,
            -376.730313462 / (ring.beta[0] * ring.gamma[0] ** 2),
            rf_station,
        )

        self.total_induced_voltage_ORG = TotalInducedVoltage(
            beam, profile1, [steps, dir_space_charge]
        )
        profile_container = ProfileContainer()
        profile_container.add_profile(profile1)
        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(steps)
        induced_voltage_container.add_induced_voltage(dir_space_charge)

        self.obj = TotalInducedVoltageNew(
            beam=beam,
            profile_container=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )

    def test_something(self):
        pass

    def test_track(self):
        self.obj.track()

    def test__induced_voltage_sum(self):
        self.obj._induced_voltage_sum()

    def test__get_compressed_wake_kernel(self):
        self.obj._get_compressed_wake_kernel()


if __name__ == "__main__":
    unittest.main()
