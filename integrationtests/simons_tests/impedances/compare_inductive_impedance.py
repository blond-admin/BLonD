# Copyright 2014-2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Comparison of 'EX_02_Main_long_ps_booster.py'."""

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, m_p

resources = (
    "/home/slauber/PycharmProjects/deleteme/blonder/legacy"
    "/__EXAMPLES/input_files"
)
ind_volt_freq_active = True
steps_active = True
dir_space_charge_active = True


class _CompareBlond23:
    def __init__(self):
        # SIMULATION PARAMETERS -------------------------------------------------------

        # Beam parameters
        self.n_particles = 1e11
        self.n_macroparticles = int(5e5)
        self.sigma_dt = 180e-9 / 4  # [s]
        self.kin_beam_energy = 1.4e9  # [eV]

        # Machine and RF parameters
        self.radius = 25
        self.gamma_transition = 4.4  # [1]
        self.circumference = 2 * np.pi * self.radius  # [m]

        # Tracking details
        self.n_turns = 2
        self.n_turns_between_two_plots = 1

        # Derived parameters
        self.E_0 = m_p * c**2 / e  # [eV]
        self.tot_beam_energy = self.E_0 + self.kin_beam_energy  # [eV]
        self.sync_momentum = np.sqrt(
            self.tot_beam_energy**2 - self.E_0**2
        )  # [eV / c]
        self.momentum_compaction = 1 / self.gamma_transition**2  # [1]

        # Cavities parameters
        self.n_rf_systems = 1
        self.harmonic_number = 1
        self.voltage = 8e3  # [V]
        self.phi_offset = np.pi

        # ejection kicker
        self.Ekicker = np.loadtxt(
            resources + "/EX_02_Ekicker_1.4GeV.txt",
            skiprows=1,
            dtype=complex,
            encoding="utf-8",
            converters={
                0: lambda s: complex(
                    bytes(s, encoding="utf-8")
                    .decode("UTF-8")
                    .replace("i", "j")
                ),
                1: lambda y: complex(
                    bytes(y, encoding="utf-8")
                    .decode("UTF-8")
                    .replace("i", "j")
                ),
            },
        )

        # Finemet cavity
        F_C = np.loadtxt(
            resources + "/EX_02_Finemet.txt",
            dtype=float,
            skiprows=1,
        )

        F_C[:, 3], F_C[:, 5], F_C[:, 7] = (
            np.pi * F_C[:, 3] / 180,
            np.pi * F_C[:, 5] / 180,
            np.pi * F_C[:, 7] / 180,
        )

        Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
        Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
        self.Re_z = 13 * Re_Z
        self.Im_z = 13 * Im_Z
        self.F_z = F_C[:, 0]

    def _exec_blond2(self):
        from blond.legacy.blond2.beam.beam import Beam, Proton
        from blond.legacy.blond2.beam.distributions import bigaussian
        from blond.legacy.blond2.beam.profile import CutOptions, Profile
        from blond.legacy.blond2.impedances.impedance import (
            InducedVoltageFreq,
            InductiveImpedance,
            TotalInducedVoltage,
        )
        from blond.legacy.blond2.impedances.impedance_sources import InputTable
        from blond.legacy.blond2.input_parameters.rf_parameters import (
            RFStation,
        )
        from blond.legacy.blond2.input_parameters.ring import Ring
        from blond.legacy.blond2.trackers.tracker import RingAndRFTracker
        from blond.legacy.blond2.utils import bmath as bm

        bm.use_cpp()

        ring = Ring(
            self.circumference,
            self.momentum_compaction,
            self.sync_momentum,
            Proton(),
            self.n_turns,
        )

        RF_sct_par = RFStation(
            ring,
            [self.harmonic_number],
            [self.voltage],
            [self.phi_offset],
            self.n_rf_systems,
        )

        my_beam = Beam(ring, self.n_macroparticles, self.n_particles)

        ring_RF_section = RingAndRFTracker(RF_sct_par, my_beam)

        # DEFINE BEAM------------------------------------------------------------------
        bigaussian(ring, RF_sct_par, my_beam, self.sigma_dt, seed=1)
        dt_init = my_beam.dt.copy()
        dE_init = my_beam.dE.copy()
        # DEFINE SLICES----------------------------------------------------------------
        slice_beam = Profile(
            my_beam,
            CutOptions(
                cut_left=-5.72984173562e-7,
                cut_right=5.72984173562e-7,
                n_slices=10000,
            ),
        )

        Ekicker_table = InputTable(
            self.Ekicker[:, 0].real,
            self.Ekicker[:, 1].real,
            self.Ekicker[:, 1].imag,
        )

        F_C_table = InputTable(self.F_z, self.Re_z, self.Im_z)

        # steps
        steps = InductiveImpedance(
            my_beam,
            slice_beam,
            34.6669349520904 / 10e9 * ring.f_rev,
            RF_sct_par,
            deriv_mode="diff",
        )
        # direct space charge
        dir_space_charge = InductiveImpedance(
            my_beam,
            slice_beam,
            -376.730313462 / (ring.beta[0] * ring.gamma[0] ** 2),
            RF_sct_par,
        )

        # INDUCED VOLTAGE FROM IMPEDANCE------------------------------------------------

        imp_list = [Ekicker_table, F_C_table]

        ind_volt_freq = InducedVoltageFreq(
            my_beam, slice_beam, imp_list, frequency_resolution=2e5
        )

        induced_voltage_list = []
        if ind_volt_freq_active:
            induced_voltage_list.append(ind_volt_freq)
        if steps_active:
            induced_voltage_list.append(steps)
        if dir_space_charge_active:
            induced_voltage_list.append(dir_space_charge)
        total_induced_voltage = TotalInducedVoltage(
            my_beam, slice_beam, induced_voltage_list
        )

        map_ = [total_induced_voltage] + [ring_RF_section] + [slice_beam]
        t0 = time.time()
        for i in range(1, self.n_turns + 1):
            print(i)

            for m in map_:
                m.track()
        t1 = time.time()
        print("Runtime BLonD2", t1 - t0, "s")
        return (
            total_induced_voltage.induced_voltage,
            slice_beam.n_macroparticles,
            dt_init,
            dE_init,
        )

    def _exec_blond3(self, dt_init, dE_init):
        from blond import (
            Beam,
            ConstantMagneticCycle,
            DriftSimple,
            ImpedanceTableFreq,
            InductiveImpedance,
            InductiveImpedanceSolver,
            PeriodicFreqSolver,
            Ring,
            Simulation,
            SingleHarmonicRFStation,
            StaticProfile,
            WakeField,
            backend,
            proton,
        )

        backend.set_specials("cpp")

        ring = Ring(circumference=self.circumference)
        beam = Beam(intensity=self.n_particles, particle_type=proton)
        cycle = ConstantMagneticCycle(
            reference_particle=beam.reference.particle_type,
            value=self.sync_momentum,
            in_unit="momentum",
        )
        beam.setup_beam(
            dt=dt_init,
            dE=dE_init,
            reference_total_energy=cycle.get_total_energy_init(),
        )
        rf_station = SingleHarmonicRFStation(
            voltage=self.voltage,
            phi_rf=self.phi_offset,
            harmonic=self.harmonic_number,
        )
        drift = DriftSimple(
            orbit_length=ring.circumference,
            momentum_compaction_factor=self.momentum_compaction,
        )
        profile = StaticProfile(
            cut_left=-5.72984173562e-7,
            cut_right=5.72984173562e-7,
            n_bins=10000,
        )

        ind_volt_freq = WakeField(
            sources=(
                ImpedanceTableFreq(
                    freq_x=self.Ekicker[:, 0].real, freq_y=self.Ekicker[:, 1]
                ),
                ImpedanceTableFreq(
                    freq_x=self.F_z, freq_y=self.Re_z + 1j * self.Im_z
                ),
            ),
            solver=PeriodicFreqSolver(t_periodicity=1 / 2e5),
            profile=profile,
        )

        f_rev = 1 / cycle.get_t_rev_init(ring.circumference)
        steps = WakeField(
            sources=(
                InductiveImpedance(Z_over_n=34.6669349520904 / 10e9 * f_rev),
            ),
            solver=InductiveImpedanceSolver(),
            profile=profile,
        )

        dir_space_charge = WakeField(
            sources=(
                InductiveImpedance(
                    Z_over_n=-376.730313462
                    / (beam.reference.beta * beam.reference.gamma**2)
                ),
            ),
            solver=InductiveImpedanceSolver(),
            profile=profile,
        )

        ind_volt_freq.track_profile = False
        steps.track_profile = False
        dir_space_charge.track_profile = False

        ring.add_elements(
            (
                ind_volt_freq if ind_volt_freq_active else None,
                steps if steps_active else None,
                dir_space_charge if dir_space_charge_active else None,
                rf_station,
                drift,
                profile,
            )
        )

        simulatuion = Simulation(ring=ring, magnetic_cycle=cycle)
        simulatuion.print_one_turn_execution_order()

        profile.track(beam=beam)

        # simulatuion.profiling(beams=deepcopy(beam), n_turns=10)
        t0 = time.time()
        simulatuion.run_simulation(beams=beam, n_turns=2)
        t1 = time.time()
        print("Runtime BLonD3", t1 - t0, "s")
        total_voltage = 0
        if ind_volt_freq_active:
            total_voltage += ind_volt_freq.induced_voltage
        if steps_active:
            total_voltage += steps.induced_voltage
        if dir_space_charge_active:
            total_voltage += dir_space_charge.induced_voltage
        return total_voltage, profile.hist_y

    def execute(self):
        induced_voltage_blond2, hist_y_blond2, dt_init, dE_init = (
            self._exec_blond2()
        )
        induced_voltage_blond3, hist_y_blond3 = self._exec_blond3(
            dt_init, dE_init
        )
        plt.subplot(2, 1, 1)
        plt.plot(hist_y_blond2, label="blond2")
        plt.plot(hist_y_blond3, "--", label="blond3")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(induced_voltage_blond2, label="induced_voltage_blond2")
        plt.plot(induced_voltage_blond3, "--", label="induced_voltage_blond3")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    comparison_of_inductive_impedance = _CompareBlond23()
    comparison_of_inductive_impedance.execute()
