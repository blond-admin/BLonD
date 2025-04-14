import unittest

import matplotlib.pyplot as plt
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
from blond.impedances.impedance import InducedVoltageFreq
from blond.impedances.impedance import InducedVoltageTime
from blond.impedances.impedance import (
    InductiveImpedance,
    TotalInducedVoltage,
)
from blond.impedances.impedance_sources import Resonators
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring


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
        for i, indced_voltage_i in enumerate(self.induced_voltage_container):
            for j, indced_voltage_j in enumerate(self.induced_voltage_container):
                if i != j:
                    assert indced_voltage_i is not indced_voltage_j


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
            CutOptions(cut_left=np.min(dt), cut_right=np.max(dt), n_slices=64),
        )
        profile1.track()
        self.profile2 = Profile(
            beam,
            CutOptions(cut_left=sigma_dt, cut_right=3 * sigma_dt, n_slices=64),
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
        profile_container = ProfileContainer()
        profile_container.add_profile(profile1)
        # profile_container.add_profile(profile2)
        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(steps)
        induced_voltage_container.add_induced_voltage(dir_space_charge)

        self.total_induced_voltage_NEW = TotalInducedVoltageNew(
            beam=beam,
            profile_container=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )

        self.total_induced_voltage_ORG = TotalInducedVoltage(
            beam, profile1, [steps, dir_space_charge]
        )

    def test___init__(self):
        pass

    def test_track(self):
        self.total_induced_voltage_NEW.track()

    def test__induced_voltage_sum_single_profile(self):
        self.total_induced_voltage_NEW._induced_voltage_sum()
        self.total_induced_voltage_ORG.induced_voltage_sum()
        DEV_DEBUG = True
        if DEV_DEBUG:
            plt.subplot(4, 1, 1)
            plt.plot(
                self.total_induced_voltage_NEW._profile_container._profiles[
                    0
                ].n_macroparticles,
                "-x",
            )
            plt.subplot(4, 1, 2)
            plt.plot(self.total_induced_voltage_NEW._compressed_wake_kernel)
            plt.axvline(64)
            plt.subplot(4, 1, 3)
            plt.plot(
                self.total_induced_voltage_NEW._profile_container._profiles[0].wake[1:],
                label="TotalInducedVoltageNew",
            )  # NOQA
            # `wake` is hidden variable of `_induced_voltage_sum`
            # plt.twinx()
            plt.plot(
                self.total_induced_voltage_ORG.induced_voltage[1:],
                "--",
                label="total_induced_voltage_ORG",
            )
            plt.subplot(4, 1, 4)
            plt.plot(
                self.total_induced_voltage_NEW._profile_container._profiles[0].wake[1:]
                - self.total_induced_voltage_ORG.induced_voltage[1:],
                label="TotalInducedVoltageNew",
            )
            """compressed_wake = fftconvolve(
                self.total_induced_voltage_NEW._compressed_wake_kernel[:],
                self.total_induced_voltage_NEW._profile_container._profiles[0].n_macroparticles[:],
                mode="same",
            )
            plt.legend(loc="upper right")
            #plt.twinx()
            plt.plot(compressed_wake[:],
                     "--",
                     label="fftconvolve",
                     )"""
            plt.legend(loc="upper left")
            plt.show()
        np.testing.assert_allclose(
            self.total_induced_voltage_NEW._profile_container._profiles[0].wake[1:],
            self.total_induced_voltage_ORG.induced_voltage[1:],
            atol=5e-2 * np.max(np.abs(self.total_induced_voltage_ORG.induced_voltage)),
        )

    def test__induced_voltage_sum_multi_profile(self):
        print("--" * 20)
        N_PROFILES = 3
        EMPTY_BUCKETS = 10
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
        for i in range(N_PROFILES):
            _dt = (
                np.random.randn(1001) * sigma_dt / 10
                + ((EMPTY_BUCKETS + 1) * i) * sigma_dt
            )
            _dE = np.random.randn(len(_dt)) / sigma_dt / 10
            if i == 0:
                dE = _dE
                dt = _dt
            else:
                dt = np.concatenate((dt, _dt))  # NOQA
                dE = np.concatenate((dE, _dE))  # NOQA

        beam = Beam(ring, len(dt), 1e11, dt=dt, dE=dE)
        profiles = []
        for i in range(N_PROFILES):
            _profile = Profile(
                beam,
                CutOptions(
                    cut_left=(((EMPTY_BUCKETS + 1) * i) - 0.5) * sigma_dt,
                    cut_right=(((EMPTY_BUCKETS + 1) * i) + 0.5) * sigma_dt,
                    n_slices=64,
                ),
            )
            _profile.track()
            profiles.append(_profile)
        del _profile
        # DEFINE SLICES----------------------------------------------------------------
        cut_left = profiles[0].cut_left
        cut_right = profiles[-1].cut_right
        width = cut_right - cut_left
        n_bins = int(round(width / profiles[0].bin_width))
        profile_full = Profile(
            beam,
            CutOptions(
                cut_left=profiles[0].cut_left,
                cut_right=profiles[-1].cut_right,
                n_slices=n_bins,
            ),
        )
        steps = InductiveImpedance(
            beam,
            profile_full,
            34.6669349520904 / 10e9 * ring.f_rev,
            rf_station,
            deriv_mode="diff",
        )

        # direct space charge
        dir_space_charge = InductiveImpedance(
            beam,
            profile_full,
            -376.730313462 / (ring.beta[0] * ring.gamma[0] ** 2),
            rf_station,
        )
        profile_container = ProfileContainer()
        for profile in profiles:
            profile_container.add_profile(profile)
        induced_voltage_freq_resonators = InducedVoltageTime(
            beam=beam,
            profile=profile_full,
            wake_source_list=[Resonators([4.5e6], [100.222e6], [200])],
        )

        induced_voltage_freq_resonators.process()
        OPTION = 1
        if OPTION == 0:
            induced_voltage_list = [steps, dir_space_charge]
        elif OPTION == 1:
            induced_voltage_list = [induced_voltage_freq_resonators]
        else:
            raise ValueError(f"{OPTION=}")
        induced_voltage_container = InducedVoltageContainer()
        for induced_voltage_tmp in induced_voltage_list:
            induced_voltage_container.add_induced_voltage(induced_voltage_tmp)
        ax = plt.subplot(3, 1, 2)
        plt.cla()
        self.total_induced_voltage_NEW = TotalInducedVoltageNew(
            beam=beam,
            profile_container=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )

        self.total_induced_voltage_ORG = TotalInducedVoltage(
            beam, profile_full, induced_voltage_list
        )

        import time
        total_induced_voltage_NEW = 0.
        total_induced_voltage_ORG = 0.
        for i in range(100):
            t0 = time.time()
            self.total_induced_voltage_NEW._induced_voltage_sum()
            t1 = time.time()
            total_induced_voltage_NEW += t1 - t0

            t0 = time.time()
            self.total_induced_voltage_ORG.induced_voltage_sum()
            t1 = time.time()
            total_induced_voltage_ORG += t1 - t0
        print("total_induced_voltage_NEW", total_induced_voltage_NEW)

        print("total_induced_voltage_ORG", total_induced_voltage_ORG)

        ax = plt.subplot(3, 1, 1)
        for profile in profiles:
            plt.plot(
                profile.bin_centers,
                profile.n_macroparticles,
            )

        plt.subplot(3, 1, 2)
        # plt.plot(
        #    profile_full.bin_centers,
        #    self.total_induced_voltage_ORG.induced_voltage,
        #    label="total_induced_voltage_ORG.induced_voltage"
        # )
        w = profile_full.cut_right - profile_full.cut_left
        profile_full_bin_centers = np.linspace(
            profile_full.cut_left - w ,
            profile_full.cut_right + w ,
            (3*profile_full.number_of_bins) + 1,
        )
        dt = profile_full_bin_centers[1] - profile_full_bin_centers[0]
        profile_full_bin_centers = profile_full_bin_centers[:-1] + dt / 2
        induced_voltage_freq_resonators.sum_wakes(profile_full_bin_centers)
        wake_kernel = induced_voltage_freq_resonators.total_wake
        wake = np.convolve(profile_full.n_macroparticles, wake_kernel, mode="same")
        wake_dt = (np.arange(len(wake)) - 800) * dt
        # plt.plot((np.arange(len(wake_kernel))) * profile_full.bin_width,
        #          wake_kernel)
        plt.legend()
        plt.subplot(3, 1, 3, sharex=ax)

        plt.plot(wake_dt, wake, label="wake np.convolve")
        ymax = max(plt.ylim())
        plt.ylim(-ymax, ymax)
        plt.legend(loc="upper right")

        # plt.twinx()

        for profile in self.total_induced_voltage_NEW._profile_container:
            plt.plot(
                profile.bin_centers,
                profile.wake,
                "--",
                c="C1",
                label="total_induced_voltage_NEW",
            )
        plt.legend(loc="lower left")
        ymax = max(plt.ylim())
        plt.ylim(-ymax, ymax)

        plt.show()


if __name__ == "__main__":
    unittest.main()
