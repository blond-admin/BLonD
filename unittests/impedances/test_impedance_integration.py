import time
import unittest
from typing import Optional

import matplotlib.pyplot as plt
def yolo(*args, **kwargs):
    raise Exception
import numpy as np
from parameterized import parameterized

from blond.beam.profile import Profile, CutOptions
from blond.beam.profilecontainer import (
    TotalInducedVoltageNew,
    EquiSpacedProfiles,
    InducedVoltageContainer,
)
from blond.impedances.impedance import (
    TotalInducedVoltage,
    InducedVoltageFreq,
    InducedVoltageTime,
    InductiveImpedance,
    InducedVoltageResonator,
)
from blond.impedances.impedance_sources import (
    InputTableTimeDomain,
    InputTableFrequencyDomain,
    Resonators,
    TravelingWaveCavity,
    ResistiveWall,
    CoherentSynchrotronRadiation,
)
np.random.seed(1)
r50_a = np.random.rand(50)
r50_b = np.random.rand(50)

class MyTestCase(unittest.TestCase):
    def setUp(self):
        import sys

        sys.path.append("/home/slauber/PycharmProjects/BLonD/unittests")
        from example_simulation import ExampleSimulation

        self.sim = ExampleSimulation()

    def get_impedance_source(self, mode: str):
        t_max = self.sim.profile.cut_right

        if mode == "InputTableTimeDomain":
            time_array = np.linspace(-t_max, 2 * t_max)
            wake = np.exp(-time_array / t_max) * np.sin(time_array / t_max * 10)
            wake[time_array < 0] = 0
            wake[np.argmin(np.abs(time_array))] = 0.5 * wake.max()
            impedance_source = InputTableTimeDomain(
                time_array=time_array,
                wake=wake,
            )
        elif mode == "InputTableFrequencyDomain":
            impedance_source = InputTableFrequencyDomain(
                frequency_array=np.linspace(0, 10e9, 50),
                Re_Z_array=r50_a,
                Im_Z_array=r50_b,
            )
            impedance_source.t_periodicity = 1 / 11e3
        elif mode == "Resonators":
            impedance_source = Resonators(
                R_S=[1, 2, 3],
                frequency_R=[1, 2, 3],
                Q=[1, 2, 3],
            )
        elif mode == "TravelingWaveCavity":
            impedance_source = TravelingWaveCavity(
                R_S=[1, 2, 3],
                frequency_R=[1, 2, 3],
                a_factor=[1, 2, 3],
            )
            impedance_source.imped_calc(np.linspace(0, 10e9, 50))
            # object doesn't initialize impedance for some reason
        elif mode == "ResistiveWall":
            impedance_source = ResistiveWall(
                pipe_radius=0.1,
                pipe_length=20,
                resistivity=3e6,
                # conductivity=conductivity,
            )
            impedance_source.imped_calc(
                frequency_array=np.linspace(0, 20039436719.676308, 50)
            )  # object
            # doesnt initialize impedance
        elif mode == "CoherentSynchrotronRadiation":
            impedance_source = CoherentSynchrotronRadiation(
                r_bend=1.2,
                # gamma=gamma,
                # chamber_height=chamber_height,
            )
        else:
            raise ValueError(mode)
        return impedance_source

    def get_induced_voltage(
        self,
        mode: str,
        mode_impedance: Optional[str] = None,
        profile: Optional[Profile] = None,
    ):
        if profile is None:
            profile = self.sim.profile
        if mode == "InducedVoltageTime":
            wake_source_list = [self.get_impedance_source(mode=mode_impedance)]
            induced_voltage = InducedVoltageTime(
                beam=self.sim.beam,
                profile=profile,
                wake_source_list=wake_source_list,
                # wake_length=wake_length,
                # multi_turn_wake=multi_turn_wake,
                # rf_station=rf_station,
                # mtw_mode=mtw_mode,
                # use_regular_fft=use_regular_fft,
            )
        elif mode == "InducedVoltageFreq":
            impedance_source_list = [self.get_impedance_source(mode=mode_impedance)]
            induced_voltage = InducedVoltageFreq(
                beam=self.sim.beam,
                profile=profile,
                impedance_source_list=impedance_source_list,
                #frequency_resolution=11.2455e3,
                # multi_turn_wake=multi_turn_wake,
                # front_wake_length=front_wake_length,
                # rf_station=rf_station,
                # mtw_mode=mtw_mode,
                # use_regular_fft=use_regular_fft,
            )
        elif mode == "InductiveImpedance":
            induced_voltage = InductiveImpedance(
                beam=self.sim.beam,
                profile=profile,
                Z_over_n=[100] * self.sim.rf_station.n_turns,
                rf_station=self.sim.rf_station,
                # deriv_mode=deriv_mode,
            )
        elif mode == "InducedVoltageResonator":
            resonators = self.get_impedance_source(mode=Resonators)
            induced_voltage = InducedVoltageResonator(
                beam=self.sim.beam,
                profile=profile,
                resonators=resonators,
                # time_array=time_array,
            )
        else:
            raise ValueError(mode)
        return induced_voltage

    @parameterized.expand(
        (
            # InducedVoltageTime is disallowed because resampling of wake in
            # time-domain could lead to a lot of problems.
            # Anyway is after all only an alternative calculation method of InducedVoltageFreq
            # ("InducedVoltageTime", InputTableTimeDomain),
            # ("InducedVoltageTime", InputTableFrequencyDomain),
            # ("InducedVoltageTime", Resonators),
            # ("InducedVoltageTime", TravelingWaveCavity),
            # ("InducedVoltageTime", ResistiveWall),
            # ("InducedVoltageTime", CoherentSynchrotronRadiation),
            ("InducedVoltageFreq", "InputTableTimeDomain"),
            ("InducedVoltageFreq", "InputTableFrequencyDomain"),
            ("InducedVoltageFreq", "Resonators"),
            ("InducedVoltageFreq", "TravelingWaveCavity"),
            ("InducedVoltageFreq", "ResistiveWall"),
            ("InducedVoltageFreq", "CoherentSynchrotronRadiation"),
            ("InductiveImpedance", None),
            # ("InducedVoltageResonator", None), # NotImplementedError intended
        )
    )
    def test_induced_voltage_sum(self, mode: str, mode_impedance: Optional[str] = None):
        induced_voltage = self.get_induced_voltage(
            mode=mode, mode_impedance=mode_impedance
        )

        profile_container = EquiSpacedProfiles()
        profile_container.add_profile(profile=self.sim.profile)

        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(induced_voltage=induced_voltage)
        induced_voltage_new = TotalInducedVoltageNew(
            beam=self.sim.beam,
            equi_spaced_profiles=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )
        induced_voltage_org = TotalInducedVoltage(
            beam=self.sim.beam,
            profile=self.sim.profile,
            induced_voltage_list=[induced_voltage],
        )
        t0 = time.time()
        induced_voltage_new.induced_voltage_sum()
        t1 = time.time()
        print(f"Runtime {t1-t0} s (new)")
        t0 = time.time()
        induced_voltage_org.induced_voltage_sum()
        t1 = time.time()
        print(f"Runtime {t1-t0} s")
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.figure(2)
            induced_voltage.dev_plot()
            induced_voltage_new.dev_plot()
            plt.figure(1)
            plt.clf()
            plt.subplot(4, 1, 1)
            plt.suptitle(f"{mode} {mode_impedance}")
            plt.plot(
                induced_voltage_new.induced_voltage, label="induced_voltage_new", c="C0"
            )
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.plot(
                induced_voltage_org.induced_voltage, label="induced_voltage_org", c="C1"
            )
            plt.legend()
            plt.subplot(4, 1, 3)
            plt.plot(
                induced_voltage_new.induced_voltage, label="induced_voltage_new", c="C0"
            )
            plt.plot(
                induced_voltage_org.induced_voltage, label="induced_voltage_org", c="C1"
            )
            plt.legend()
            plt.subplot(4, 1, 4)
            plt.plot(
                induced_voltage_new.induced_voltage[1:-1]
                - induced_voltage_org.induced_voltage[1:-1],
                label="residual",
                c="C0",
            )

            plt.tight_layout()
            plt.show()

        np.testing.assert_allclose(
            induced_voltage_new.induced_voltage[1:-1], induced_voltage_org.induced_voltage[1:-1]
        )

    @parameterized.expand(
        (
            # InducedVoltageTime is disallowed because resampling of wake in
            # time-domain could lead to a lot of problems.
            # Anyway is after all only an alternative calculation method of InducedVoltageFreq
            # ("InducedVoltageTime", InputTableTimeDomain),
            # ("InducedVoltageTime", InputTableFrequencyDomain),
            # ("InducedVoltageTime", Resonators),
            # ("InducedVoltageTime", TravelingWaveCavity),
            # ("InducedVoltageTime", ResistiveWall),
            # ("InducedVoltageTime", CoherentSynchrotronRadiation),
            ("InducedVoltageFreq", "InputTableTimeDomain"),
            ("InducedVoltageFreq", "InputTableFrequencyDomain"),
            ("InducedVoltageFreq", "Resonators"),
            ("InducedVoltageFreq", "TravelingWaveCavity"),
            ("InducedVoltageFreq", "ResistiveWall"),
            ("InducedVoltageFreq", "CoherentSynchrotronRadiation"),
            ("InductiveImpedance", None),
            # ("InducedVoltageResonator", None), # NotImplementedError intended
        )
    )
    def test_induced_voltage_sum_multi_profile(
        self, mode: str, mode_impedance: Optional[str] = None
    ):
        profile1 = self.sim.profile
        offset = 2 * profile1.bin_width * profile1.number_of_bins
        profile2 = Profile(
            self.sim.beam,
            CutOptions(
                n_slices=100,
                cut_left=0 + offset,
                cut_right=self.sim.rf_station.t_rev[0]
                / self.sim.rf_station.harmonic[0, 0]
                + offset,
            ),
        )
        profile3 = Profile(
            self.sim.beam,
            CutOptions(
                n_slices=100,
                cut_left=0 + offset * 2,
                cut_right=self.sim.rf_station.t_rev[0]
                / self.sim.rf_station.harmonic[0, 0]
                + offset * 2,
            ),
        )
        profile_entire = Profile(
            self.sim.beam,
            CutOptions(
                n_slices=900,
                cut_left=0,
                cut_right=self.sim.rf_station.t_rev[0]
                / self.sim.rf_station.harmonic[0, 0]
                + offset * 4,
            ),
        )
        np.random.seed(1)
        idxs = np.random.randint(
            0, len(self.sim.beam.dE) // 2, len(self.sim.beam.dE) // 2
        )
        self.sim.beam.dt[idxs] += offset
        np.random.seed(1)
        idxs = np.random.randint(
            len(self.sim.beam.dE) // 2 + 1,
            len(self.sim.beam.dE) - 1,
            len(self.sim.beam.dE) // 2,
        )
        self.sim.beam.dt[idxs] += 2 * offset

        profile1.track()
        profile2.track()
        profile3.track()
        profile_entire.track()
        inspect_profiles = False
        if inspect_profiles:
            plt.figure()
            plt.plot(profile1.n_macroparticles)
            plt.plot(profile2.n_macroparticles)
            plt.plot(profile3.n_macroparticles)
            plt.plot(profile_entire.n_macroparticles, "--")
            plt.show()

        profile_container = EquiSpacedProfiles()
        profile_container.add_profile(profile=profile1)
        profile_container.add_profile(profile=profile2)
        profile_container.add_profile(profile=profile3)

        induced_voltage_new = self.get_induced_voltage(
            mode=mode, mode_impedance=mode_impedance, profile=profile_entire
        )
        induced_voltage_entire = self.get_induced_voltage(
            mode=mode,
            mode_impedance=mode_impedance,
            profile=profile_entire,
        )

        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(
            induced_voltage=induced_voltage_new
        )
        induced_voltage_new = TotalInducedVoltageNew(
            beam=self.sim.beam,
            equi_spaced_profiles=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )
        induced_voltage_org = TotalInducedVoltage(
            beam=self.sim.beam,
            profile=profile_entire,
            induced_voltage_list=[induced_voltage_entire],
        )
        t0 = time.time()
        induced_voltage_new.induced_voltage_sum()
        t1 = time.time()
        print(f"Runtime {t1-t0} s (new)")
        t0 = time.time()
        induced_voltage_org.induced_voltage_sum()
        t1 = time.time()
        print(f"Runtime {t1-t0} s")
        DEV_DEBUG = False
        if DEV_DEBUG:
            plt.figure(2)
            induced_voltage_new.dev_plot()
            induced_voltage_entire.dev_plot()
            for subplot_i in range(3):
                plt.subplot(4, 1, 1 + subplot_i)
                plt.legend()
            plt.figure(1)
            plt.clf()
            plt.subplot(4, 1, 1)
            plt.suptitle(f"{mode} {mode_impedance}")
            plt.plot(
                induced_voltage_new.entire_induced_voltage,
                label="induced_voltage_new",
                c="C0",
            )
            plt.legend()
            plt.subplot(4, 1, 2)
            plt.plot(
                induced_voltage_org.induced_voltage, label="induced_voltage_org", c="C1"
            )
            plt.legend()
            plt.subplot(4, 1, 3)
            plt.plot(
                induced_voltage_new.entire_induced_voltage,
                label="induced_voltage_new",
                c="C0",
            )
            plt.plot(
                induced_voltage_org.induced_voltage,
                "--",
                label="induced_voltage_org",
                c="C1",
            )
            plt.legend()
            plt.subplot(4, 1, 4)

            plt.tight_layout()

        for profile_i in range(3):
            start = profile_i * 200
            stop = start + 100
            if mode == "InductiveImpedance":
                start += 1
                stop -= 1
            if mode == "InductiveImpedance":
                sel = slice(1, 99)
            else:
                sel = slice(0,100)
            DEV_DEBUG2 = False
            if DEV_DEBUG2:
                plt.figure(11)
                plt.subplot(2, 1, 1)
                artists = plt.plot(induced_voltage_new.get_induced_voltage(
                    profile_i=profile_i)[sel],)
                color = artists[0].get_color()
                plt.plot(induced_voltage_org.induced_voltage[
                         start:stop], "--")
                plt.subplot(2, 1, 2)
                plt.plot(
                    induced_voltage_org.induced_voltage[start:stop]
                    - induced_voltage_new.get_induced_voltage(
                        profile_i=profile_i)[sel],
                    color=color
                )
                plt.show()

            np.testing.assert_allclose(
                induced_voltage_new.get_induced_voltage(profile_i=profile_i)[sel],
                induced_voltage_org.induced_voltage[start:stop],
                atol=1e-12,
                # must be atol, because most data is around zero
            )



if __name__ == "__main__":
    unittest.main()
