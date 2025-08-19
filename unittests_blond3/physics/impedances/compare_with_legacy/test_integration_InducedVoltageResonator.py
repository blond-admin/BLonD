import unittest

import numpy as np
from scipy.constants import c, e, m_p
import matplotlib.pyplot as plt

from blond3 import (
    Ring,
    Beam,
    proton,
    DriftSimple,
    SingleHarmonicCavity,
    StaticProfile,
    WakeField,
    Simulation,
    ConstantMagneticCycle,
)
from blond3._core.backends.backend import backend, Numpy64Bit
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import (
    PeriodicFreqSolver, AnalyticSingleTurnResonatorSolver,
)

from .test_integration_InducedVoltageFreq import f_res, Q_factor, R_shunt

DEV_PLOT = True


class Blond2:
    def __init__(self):
        from blond.beam.beam import Beam, Proton
        from blond.beam.distributions import bigaussian
        from blond.beam.profile import CutOptions, Profile
        from blond.impedances.impedance import (
            InducedVoltageFreq,
            InducedVoltageResonator,
            InducedVoltageTime,
            TotalInducedVoltage,
        )
        from blond.impedances.impedance_sources import Resonators
        from blond.input_parameters.rf_parameters import RFStation
        from blond.input_parameters.ring import Ring

        for solver in (
            InducedVoltageTime,
            # InducedVoltageFreq,
            # InducedVoltageResonator,
        ):
            ring = Ring(6911.56, 1 / (1 / np.sqrt(0.00192)) ** 2, 25.92e9, Proton(), 10)
            rf_station = RFStation(ring, [4620], [0.9e6], [0.0], 1)
            beam = Beam(ring, 1e6, 1e10)
            bigaussian(ring, rf_station, beam, 2e-9 / 4, seed=1)
            self.dt = beam.dt
            self.dE = beam.dE

            cut_options = CutOptions(
                cut_left=0,
                cut_right=2 * np.pi,
                n_slices=64,
                rf_station=rf_station,
                cuts_unit="rad",
            )
            profile = Profile(beam, cut_options)
            self.profile = profile

            profile.track()
            R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
            resonator = Resonators(R_shunt, f_res, Q_factor)
            self.resonator = resonator

            if solver == InducedVoltageTime:
                ind_volt = InducedVoltageTime(beam, profile, [resonator])
            elif solver == InducedVoltageFreq:
                ind_volt = InducedVoltageFreq(beam, profile, [resonator], 1e5)
            elif solver == InducedVoltageResonator:
                ind_volt = InducedVoltageResonator(beam, profile, resonator)
            else:
                raise Exception
            tot_vol = TotalInducedVoltage(beam, profile, [ind_volt])

            tot_vol.induced_voltage_sum()
            self.induced_voltage = tot_vol.induced_voltage

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
    def __init__(self):
        blond2 = Blond2()
        self.blond2 = blond2

        ring = Ring(circumference=6911.56)
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )
        cavity1 = SingleHarmonicCavity()
        cavity1.voltage = 0.9e6
        cavity1.phi_rf = 0
        cavity1.harmonic = 4620
        drift = DriftSimple()
        drift.transition_gamma = 1 / (1 / np.sqrt(0.00192)) ** 2
        R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
        resonators = Resonators(
            shunt_impedances=R_shunt,
            center_frequencies=f_res,
            quality_factors=Q_factor,
        )

        beam = Beam(n_particles=1e10, particle_type=proton)
        beam.setup_beam(dt=blond2.dt, dE=blond2.dE)
        profile.track(beam)

        wake = WakeField(
            sources=(resonators,),
            solver=AnalyticSingleTurnResonatorSolver(),
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
        a = np.asarray(arr_a)
        b = np.asarray(arr_b)
        diff = a - b
        den = max(np.linalg.norm(a), np.linalg.norm(b), atol / rtol if rtol > 0 else 1.0)
        return (np.linalg.norm(diff) <= rtol * den + atol * np.sqrt(diff.size))

    def test___init__(self):
        if DEV_PLOT:
            plt.show()
        plt.plot(self.blond3.blond2.induced_voltage)
        plt.plot(self.blond3.induced_voltage, ls="--")
        plt.show()
        # plt.plot(self.blond3.blond2.induced_voltage)
        # plt.plot(self.blond3.induced_voltage, ls="--")
        # plt.xlim(390*2, 395*2)
        # plt.ylim(-1e-7, 1e-7)
        # plt.show()

        assert self.close_in_norm(self.blond3.blond2.induced_voltage, self.blond3.induced_voltage)

        # np.testing.assert_allclose(
        #     self.blond3.blond2.induced_voltage, self.blond3.induced_voltage, rtol=1e-6
        # )


