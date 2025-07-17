import unittest

import matplotlib.pyplot as plt
import numpy as np

from blond3 import (
    Beam,
    Ring,
    Simulation,
    proton,
    ConstantMagneticCycle,
    StaticProfile,
    SingleHarmonicCavity,
    DriftSimple,
    WakeField,
)
from blond3._core.backends.backend import backend, Numpy64Bit
from blond3.physics.impedances.sources import Resonators
from blond3.physics.impedances.solvers import PeriodicFreqSolver

R_shunt = np.array(
    [
        388000.0,
        600.0,
        69800.0,
        87200.0,
        20400.0,
        109000.0,
        9100.0,
        1800.0,
        7500.0,
        2600.0,
        28000.0,
        29000.0,
        120000.0,
        79000.0,
        48000.0,
        14500.0,
        5000.0,
        10500.0,
        12500.0,
        10000.0,
        43000.0,
        45000.0,
        11500.0,
        4500.0,
        6900.0,
        15400.0,
        11000.0,
        4700.0,
        8500.0,
        17000.0,
        5000.0,
        633000.0,
        499000.0,
        364000.0,
        177000.0,
        61400.0,
        29900.0,
        379000.0,
        243000.0,
        17400.0,
        588000.0,
        61000.0,
        771000.0,
        187000.0,
        814000.0,
        281000.0,
        134000.0,
        42000.0,
        68000.0,
        52000.0,
        45000.0,
        41000.0,
    ]
)
f_res = np.array(
    [
        629000000.0,
        840000000.0,
        1066000000.0,
        1076000000.0,
        1608000000.0,
        1884000000.0,
        2218000000.0,
        2533000000.0,
        2547000000.0,
        2782000000.0,
        3008000000.0,
        3223000000.0,
        3284000000.0,
        3463000000.0,
        3643000000.0,
        3761000000.0,
        3900000000.0,
        4000000000.0,
        4080000000.0,
        4210000000.0,
        1076000000.0,
        1100000000.0,
        1955000000.0,
        2075000000.0000002,
        2118000000.0,
        2199000000.0,
        2576000000.0,
        2751000000.0,
        3370000000.0,
        5817000000.0,
        5817000000.0,
        1210000000.0,
        1280000000.0,
        1415000000.0,
        1415000000.0,
        1415000000.0,
        1415000000.0,
        1395000000.0,
        1401000000.0,
        1570000000.0,
        1610000000.0,
        1620000000.0,
        1861000000.0,
        1890000000.0,
        2495000000.0,
        696000000.0,
        910000000.0,
        1069000000.0,
        1078000000.0,
        1155000000.0,
        1232000000.0,
        1343000000.0,
    ]
)
Q_factor = np.array(
    [
        500.0,
        10.0,
        500.0,
        500.0,
        40.0,
        500.0,
        15.0,
        384.0,
        340.0,
        20.0,
        450.0,
        512.0,
        600.0,
        805.0,
        1040.0,
        965.0,
        50.0,
        1300.0,
        600.0,
        200.0,
        700.0,
        700.0,
        450.0,
        600.0,
        600.0,
        750.0,
        1000.0,
        500.0,
        30.0,
        1000.0,
        10.0,
        315.0,
        200.0,
        75.0,
        270.0,
        75.0,
        270.0,
        200.0,
        1100.0,
        55.0,
        980.0,
        60.0,
        810.0,
        175.0,
        1190.0,
        7400.0,
        8415.0,
        7980.0,
        7810.0,
        6660.0,
        5870.0,
        7820.0,
    ]
)

DEV_PLOT = False


class Blond2:
    def __init__(self):
        import numpy as np

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
            # InducedVoltageTime,
            InducedVoltageFreq,
            # InducedVoltageResonator,
        ):
            ring = Ring(6911.56, 1 / (1 / np.sqrt(0.00192)) ** 2, 25.92e9, Proton(), 10)
            rf_station = RFStation(ring, [4620], [0.9e6], [0.0], 1)
            beam = Beam(ring, 1001, 1e10)
            bigaussian(ring, rf_station, beam, 2e-9 / 4, seed=1)
            self.dt = beam.dt
            self.dE = beam.dE

            cut_options = CutOptions(
                cut_left=0,
                cut_right=2 * np.pi,
                n_slices=256,
                rf_station=rf_station,
                cuts_unit="rad",
            )
            profile = Profile(beam, cut_options)
            self.profile = profile

            profile.track()

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
                plt.figure(2)
                plt.plot(ind_volt.total_impedance)


class Blond3:
    def __init__(self):
        blond2 = Blond2()
        self.blond2 = blond2
        circumference = 6911.56
        ring = Ring()
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )
        cavity1 = SingleHarmonicCavity()
        cavity1.voltage = 0
        cavity1.phi_rf = 0
        cavity1.harmonic = 1
        drift = DriftSimple(effective_length=circumference)
        drift.transition_gamma = 1
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
            solver=PeriodicFreqSolver(t_periodicity=1 / 1e5, allow_next_fast_len=True),
            profile=profile,
        )
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
            plt.figure(2)
            plt.plot(wake.solver._freq_y)
        self.induced_voltage = induced_voltage


class TestBothBlonds(unittest.TestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        self.blond3 = Blond3()

    def test___init__(self):
        np.testing.assert_allclose(
            self.blond3.blond2.induced_voltage, self.blond3.induced_voltage, rtol=1e-6
        )
        if DEV_PLOT:
            plt.show()
