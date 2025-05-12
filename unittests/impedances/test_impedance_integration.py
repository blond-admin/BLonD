import unittest
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from parameterized import parameterized

from blond.beam.profilecontainer import (
    TotalInducedVoltageNew,
    ProfileContainer,
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


class MyTestCase(unittest.TestCase):
    def setUp(self):
        from ..example_simulation import ExampleSimulation

        self.sim = ExampleSimulation()

    def get_impedance_source(self, mode: str):
        t_max = self.sim.profile.cut_right

        if mode == InputTableTimeDomain:
            time_array = np.linspace(-t_max, 2 * t_max)
            wake = np.exp(-time_array / t_max) * np.sin(time_array / t_max * 10)
            wake[time_array < 0] = 0
            wake[np.argmin(np.abs(time_array))] = 0.5 * wake.max()
            impedance_source = InputTableTimeDomain(
                time_array=time_array,
                wake=wake,
            )
        elif mode == InputTableFrequencyDomain:
            impedance_source = InputTableFrequencyDomain(
                frequency_array=np.linspace(0, 10e9, 50),
                Re_Z_array=np.random.rand(50),
                Im_Z_array=np.random.rand(50),
            )
        elif mode == Resonators:
            impedance_source = Resonators(
                R_S=[1, 2, 3],
                frequency_R=[1, 2, 3],
                Q=[1, 2, 3],
            )
        elif mode == TravelingWaveCavity:
            impedance_source = TravelingWaveCavity(
                R_S=[1, 2, 3],
                frequency_R=[1, 2, 3],
                a_factor=[1, 2, 3],
            )
            impedance_source.imped_calc(np.linspace(0, 10e9, 50))  # object
            # doesnt initialize impedance
        elif mode == ResistiveWall:
            impedance_source = ResistiveWall(
                pipe_radius=0.1,
                pipe_length=20,
                resistivity=3e6,
                # conductivity=conductivity,
            )
            impedance_source.imped_calc(np.linspace(0, 10e9, 50))  # object
            # doesnt initialize impedance
        elif mode == CoherentSynchrotronRadiation:
            impedance_source = CoherentSynchrotronRadiation(
                r_bend=1.2,
                # gamma=gamma,
                # chamber_height=chamber_height,
            )
        else:
            raise ValueError(mode)
        return impedance_source

    def get_induced_voltage(self, mode: str, mode_impedance: Optional[str] = None):
        if mode == "InducedVoltageTime":
            wake_source_list = [self.get_impedance_source(mode=mode_impedance)]
            induced_voltage = InducedVoltageTime(
                beam=self.sim.beam,
                profile=self.sim.profile,
                wake_source_list=wake_source_list,
                # wake_length=wake_length,
                # multi_turn_wake=multi_turn_wake,
                # rf_station=rf_station,
                # mtw_mode=mtw_mode,
                # use_regular_fft=use_regular_fft,
            )
        elif mode == InducedVoltageFreq:
            impedance_source_list = [self.get_impedance_source(mode=mode_impedance)]
            induced_voltage = InducedVoltageFreq(
                beam=self.sim.beam,
                profile=self.sim.profile,
                impedance_source_list=impedance_source_list,
                # frequency_resolution=frequency_resolution,
                # multi_turn_wake=multi_turn_wake,
                # front_wake_length=front_wake_length,
                # rf_station=rf_station,
                # mtw_mode=mtw_mode,
                # use_regular_fft=use_regular_fft,
            )
        elif mode == InductiveImpedance:
            induced_voltage = InductiveImpedance(
                beam=self.sim.beam,
                profile=self.sim.profile,
                Z_over_n=[100] * self.sim.rf_station.n_turns,
                rf_station=self.sim.rf_station,
                # deriv_mode=deriv_mode,
            )
        elif mode == "InducedVoltageResonator":
            resonators = self.get_impedance_source(mode=Resonators)
            induced_voltage = InducedVoltageResonator(
                beam=self.sim.beam,
                profile=self.sim.profile,
                resonators=resonators,
                # time_array=time_array,
            )
        else:
            raise ValueError(mode)
        return induced_voltage

    @parameterized.expand(
        (
            # FIXME
            #  ("InducedVoltageTime", InputTableTimeDomain),
            #  ("InducedVoltageTime", InputTableFrequencyDomain),
            #  induced_voltage_org seems broken
            # ("InducedVoltageTime", Resonators), # TODO STRANGE SHIFT
            #("InducedVoltageTime", TravelingWaveCavity), # TODO STRANGE SHIFT
            # ("InducedVoltageTime", ResistiveWall), # TODO STRANGE SHIFT
            # ("InducedVoltageTime", CoherentSynchrotronRadiation),
            # NotImplementedError
            (InducedVoltageFreq, InputTableTimeDomain),  # TODO ALMOST
            #(InducedVoltageFreq, InputTableFrequencyDomain),  # TODO ALMOST
            #(InducedVoltageFreq, Resonators),  # TODO ALMOST
            # (InducedVoltageFreq, TravelingWaveCavity),  # TODO BUGGY SCALE
            #(InducedVoltageFreq, ResistiveWall),  # TODO ALMOST
            #(InducedVoltageFreq, CoherentSynchrotronRadiation),  # WORKS
            #(InductiveImpedance, None),  # WORKS
            # ("InducedVoltageResonator", None), # NotImplementedError
        )
    )
    def test_something(self, mode: str, mode_impedance: Optional[str] = None):
        induced_voltage = self.get_induced_voltage(
            mode=mode, mode_impedance=mode_impedance
        )
        profile_container = ProfileContainer()
        profile_container.add_profile(profile=self.sim.profile)
        induced_voltage_container = InducedVoltageContainer()
        induced_voltage_container.add_induced_voltage(induced_voltage=induced_voltage)
        induced_voltage_new = TotalInducedVoltageNew(
            beam=self.sim.beam,
            profile_container=profile_container,
            induced_voltage_container=induced_voltage_container,
            track_update_wake_kernel=False,
        )
        induced_voltage_org = TotalInducedVoltage(
            beam=self.sim.beam,
            profile=self.sim.profile,
            induced_voltage_list=[induced_voltage],
        )

        induced_voltage_new.induced_voltage_sum()
        induced_voltage_org.induced_voltage_sum()
        DEV_DEBUG = True
        if DEV_DEBUG:
            plt.figure(2)
            induced_voltage.dev_plot()
            induced_voltage_new.dev_plot()
            plt.figure(1)
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.suptitle(f"{mode} {mode_impedance}")
            plt.plot(
                induced_voltage_new.induced_voltage, label="induced_voltage_new", c="C0"
            )
            plt.legend()
            plt.subplot(3, 1, 2)
            plt.plot(
                induced_voltage_org.induced_voltage, label="induced_voltage_org", c="C1"
            )
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(
                induced_voltage_new.induced_voltage, label="induced_voltage_new", c="C0"
            )
            plt.plot(
                induced_voltage_org.induced_voltage, label="induced_voltage_org", c="C1"
            )
            plt.legend()
            plt.tight_layout()
            plt.show()
        np.testing.assert_allclose(
            induced_voltage_new.induced_voltage, induced_voltage_org.induced_voltage
        )


if __name__ == "__main__":
    unittest.main()
