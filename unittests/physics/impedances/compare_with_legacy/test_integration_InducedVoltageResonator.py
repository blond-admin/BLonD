import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRfStation,
    StaticProfile,
    WakeField,
    proton,
)
from blond._core.backends.backend import Numpy64Bit, backend
from blond.physics.impedances.solvers import (
    PeriodicFreqSolver,
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators

from .test_integration_InducedVoltageFreq import Q_factor, R_shunt, f_res

DEV_PLOT = False


class Blond2:
    def __init__(
        self, n_macroparticles=int(1e6), n_slices=256, bunch_length=1e-9 / 4
    ):
        from blond.legacy.blond2.beam.beam import Beam, Proton
        from blond.legacy.blond2.beam.distributions import bigaussian
        from blond.legacy.blond2.beam.profile import CutOptions, Profile
        from blond.legacy.blond2.impedances.impedance import (
            InducedVoltageFreq,
            InducedVoltageResonator,
            InducedVoltageTime,
            TotalInducedVoltage,
        )
        from blond.legacy.blond2.impedances.impedance_sources import Resonators
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring

        self.induced_voltage = []

        for solver in (
            InducedVoltageTime,
            InducedVoltageResonator,
            # InducedVoltageFreq,
        ):
            ring = Ring(
                6911.56, 0.00192, 25.92e9, Proton(), 10
            )
            rf_station = RFStation(ring, [4620], [0.9e6], [0.0], 1)
            beam = Beam(ring, n_macroparticles, 1e10)
            bigaussian(ring, rf_station, beam, bunch_length, seed=1)
            self.dt = beam.dt
            self.dE = beam.dE

            cut_options = CutOptions(
                cut_left=0,
                cut_right=2 * np.pi,
                n_slices=n_slices,
                rf_station=rf_station,
                cuts_unit="rad",
            )
            profile = Profile(beam, cut_options)
            self.profile = profile

            profile.track()
            # R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
            resonator = Resonators(R_shunt, f_res, Q_factor)
            self.resonator = resonator

            if solver == InducedVoltageTime:
                ind_volt = InducedVoltageTime(
                    beam,
                    profile,
                    [resonator],
                )
            elif solver == InducedVoltageFreq:
                ind_volt = InducedVoltageFreq(beam, profile, [resonator], 1e5)
            elif solver == InducedVoltageResonator:
                ind_volt = InducedVoltageResonator(
                    beam,
                    profile,
                    resonator,
                )
            else:
                raise Exception
            tot_vol = TotalInducedVoltage(beam, profile, [ind_volt])

            tot_vol.induced_voltage_sum()
            self.induced_voltage.append(tot_vol.induced_voltage)

            if DEV_PLOT:
                plt.figure(1)
                plt.plot(tot_vol.induced_voltage)
                if not solver == InducedVoltageResonator:
                    pass
                try:
                    plt.figure(2)
                    plt.plot(ind_volt.total_impedance)
                except ValueError:
                    pass


class Blond3:
    def __init__(
        self, n_macroparticles=int(1e6), n_slices=256, bunch_length=1e-9 / 4
    ):
        blond2 = Blond2(
            n_macroparticles=n_macroparticles,
            n_slices=n_slices,
            bunch_length=bunch_length,
        )
        self.blond2 = blond2

        ring = Ring(circumference=6911.56)
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )
        cavity1 = SingleHarmonicRfStation()
        cavity1.voltage = 0.9e6
        cavity1.phi_rf = 0
        cavity1.harmonic = 4620
        drift = DriftSimple(orbit_length=ring.circumference)
        drift.transition_gamma = 1 / (1 / np.sqrt(0.00192)) ** 2
        # R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
        resonators = Resonators(
            shunt_impedances=R_shunt,
            center_frequencies=f_res,
            quality_factors=Q_factor,
        )

        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(dt=blond2.dt, dE=blond2.dE)
        profile.track(beam)

        wake = WakeField(
            sources=(resonators,),
            solver=SingleTurnResonatorConvolutionSolver(),
            profile=profile,
        )
        wake.solver._wake_pot_vals_need_update = True
        ring.add_elements((profile, cavity1, drift, wake))
        magnetic_cycle = ConstantMagneticCycle(
            value=25.92e9,
            reference_particle=proton,
        )
        sim = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)

        induced_voltage = wake.calc_induced_voltage(beam=beam)
        if DEV_PLOT:
            plt.figure(1)
            plt.plot(induced_voltage, "--", color="r")
            try:
                plt.figure(2)
                plt.plot(wake.solver._freq_y)
            except AttributeError:
                pass
        self.induced_voltage = induced_voltage


class TestBothBlonds(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        self.blond3 = Blond3()

    def close_in_norm(self, arr_a, arr_b, rtol=1e-5, atol=1e-8):
        norm = np.inf
        a = np.asarray(arr_a)
        b = np.asarray(arr_b)
        diff = a - b
        den = max(
            np.linalg.norm(a, norm),
            np.linalg.norm(b, norm),
            atol / rtol if rtol > 0 else 1.0,
        )
        return np.linalg.norm(diff, norm) <= rtol * den + atol * np.sqrt(
            diff.size
        )

    def test_integration(self):
        n_macroparticles = int(1e6)
        n_slices = 1024  # this falls apart at low n_slices as the fftconvolve becomes inaccurate (BLonD2)
        bunch_length = 1e-9 / 8
        self.blond3 = Blond3(n_macroparticles, n_slices, bunch_length)

        DEBUG_PLOT = False
        if DEBUG_PLOT:
            plt.title(f"{n_macroparticles} {n_slices} {bunch_length}")
            plt.plot(
                self.blond3.blond2.induced_voltage[0], label="blond2 ind_volt time"
            )
            plt.plot(self.blond3.blond2.induced_voltage[1], label="blond2 ind volt res", ls=":")
            # plt.plot(self.blond3.blond2.induced_voltage[2], label="blond2 ind volt freq", ls="--")
            plt.plot(self.blond3.induced_voltage, label="blond3", ls="dashdot")
            plt.legend()
            plt.show()

        for blond2_ind_volt in self.blond3.blond2.induced_voltage:
            try:
                assert self.close_in_norm(
                    blond2_ind_volt, self.blond3.induced_voltage
                )
            except AssertionError:
                np.testing.assert_allclose(
                    blond2_ind_volt, self.blond3.induced_voltage, atol=20  # of 120000
                )

    def test_diff_params(self):
        DEBUG_MODE = False
        if DEBUG_MODE:
            n_macroparts = [int(1e4), int(1e5), int(1e6)]
            bunch_lengths = [1e-8 / 12, 1e-9 / 8,  1e-9 / 4, ]
            n_slices_lst = [64, 128, 256, 512]
        else:
            n_macroparts = [int(1e5)]
            bunch_lengths = [1e-9 / 4]
            n_slices_lst = [64]
        for mac_ind, n_macroparticles in enumerate(
            n_macroparts
        ):
            for slic_ind, n_slices in enumerate(n_slices_lst):
            # for slic_ind, n_slices in enumerate([1024]):
                # for b_ind, bunch_length in enumerate([1e-9 / 4, 1e-9, 4e-9]):
                for b_ind, bunch_length in enumerate(bunch_lengths):
                    self.blond3 = Blond3(
                        n_macroparticles, n_slices, bunch_length
                    )
                    DEBUG_PLOT = False
                    if DEBUG_PLOT:
                        plt.title(f"{n_macroparticles} {n_slices} {bunch_length}")
                        plt.plot(
                            self.blond3.blond2.induced_voltage[0],
                            label="blond2 ind_volt time",
                        )
                        plt.plot(
                            self.blond3.blond2.induced_voltage[1],
                            label="blond2 ind volt res",
                            ls=":",
                        )
                        # plt.plot(self.blond3.blond2.induced_voltage[2], label="blond2 ind volt res", ls="--")
                        plt.plot(
                            self.blond3.induced_voltage,
                            label="blond3",
                            ls="dashdot",
                        )
                        plt.legend()
                        plt.show()

                    for blond2_ind_volt in self.blond3.blond2.induced_voltage:
                        try:
                            assert self.close_in_norm(
                                blond2_ind_volt, self.blond3.induced_voltage
                            )
                        except AssertionError:
                            np.testing.assert_allclose(
                                blond2_ind_volt, self.blond3.induced_voltage, rtol=1e-2, atol=200
                            )
