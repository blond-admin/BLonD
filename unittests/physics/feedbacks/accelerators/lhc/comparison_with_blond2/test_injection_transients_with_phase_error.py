import unittest

import numpy as np
import scipy.constants as sc
from matplotlib import pyplot as plt
from tqdm import tqdm

# Accelerator
circumference = 26658.8832  # [m]
momentum = 450e9
n_bunches = 36
intensity = 1.6e11 * n_bunches
n_turns = 500
gamma_t = 53.606713
delta_f = -3480
alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor [-]

bunch_lengths = 1.2e-9

bucket_shift = 10_000
injection_phase_error = 40

energy = np.sqrt(momentum**2 + (sc.proton_mass * sc.c**2 / sc.e) ** 2)
rel_gamma = energy / (sc.proton_mass * sc.c**2 / sc.e)
rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

# The RF station
h = 35640  # Harmonic number [-]
voltage = 5e6  # RF voltage [V]
dphi = 0  # Phase modulation/offset [rad]

# Cavity Controller
G_a = 6.79e-6  # Analog FB gain [A/V]
G_d = 10  # Digital FB gain [-]
tau_loop = 650e-9  # Overall loop delay [s]
tau_a = 170e-6  # Analog FB delay [s]
tau_d = 400e-6  # Digital FB delay [s]
a_comb = 15 / 16  # Comb filter alpha [-]
Q_L = 20000  # Loaded Quality factor [-]
G_otfb = 10  # OTFB gain [-]
tau_comp = 1200e-9  # Complimentary delay in OTFB [s]

# The beam
number_of_bunches = 36  # Length of the batch [number of bunches]
bunch_intensity = 1.6e11  # Bunch intensity [p/b]
n_macroparticles = 100_000  # Number of macroparticles per bunch [-]
tau_bunch = 1.2e-9  # Bunch length [s]
bunch_spacing = 10  # Bunch spacing [number of rf buckets]
injection_energy_error = 0  # Injection energy error [eV]


class TestInjectionWithPhaseError(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def setup_blond3():
            from blond import (
                Beam,
                BiGaussian,
                ConstantMagneticCycle,
                DriftSimple,
                MultiHarmonicRFStation,
                Ring,
                Simulation,
                StaticProfile,
                backend,
                proton,
            )
            from blond.core.backends.backend import Numpy64Bit
            from blond.experimental.physics.feedbacks.accelerators.lhc.beam_feedback import (
                LHCBeamControl,
            )
            from blond.experimental.physics.feedbacks.accelerators.lhc.cavity_feedback import (
                LHCCavityLoop,
                LHCCavityLoopCommissioning,
            )
            from blond.generals.distributed.distributed_array import (
                DistributedArray,
            )

            backend.change_backend(Numpy64Bit)
            backend.set_specials("cpp")

            beam = Beam(
                intensity,
                proton,
            )

            cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

            lattice = DriftSimple(
                orbit_length=circumference,
                momentum_compaction_factor=1.0 / gamma_t / gamma_t,
            )

            cavity = MultiHarmonicRFStation(
                voltage=np.array([voltage]),
                phi_rf=np.array([dphi]),
                harmonic=np.array([h]),
                n_harmonics=1,
                main_harmonic_idx=0,
            )

            f_rf = cavity.calc_main_harmonic_omega_rf_design(
                rel_beta, lattice.orbit_length
            ) / (2 * np.pi)
            f_rev = f_rf / h
            t_rf = 1 / f_rf
            t_rev = 1 / f_rev

            profile = StaticProfile(
                cut_left=(bucket_shift - 5.5) / f_rf,
                cut_right=(bucket_shift + 6.5 + n_bunches * 10) / f_rf,
                n_bins=2**5 * (12 + n_bunches * 10),
            )

            # LHC cavity feedback
            commissioning = LHCCavityLoopCommissioning(
                G_a=6.79e-6,
                G_d=10,
                G_o=10,
                tau_a=170e-6,
                tau_d=400e-6,
                tau_o=110e-6,
            )
            cavity_control = LHCCavityLoop(
                profile=profile,
                tau_otfb=1.2e-6,
                f_c=f_rf + delta_f,
                RFFB=commissioning,
                n_pretrack=200,
            )
            cavity.attach_cavity_feedback(cavity_control)

            # LHC beam feedback
            beam_control = LHCBeamControl(
                profile,
                pl_gain=1 / (5 * t_rev) * 1,
                sl_gain=1 / (5 * t_rev) / 10 * 1,
                current_thres=0.5,
            )
            cavity.attach_beam_feedback(beam_control)

            bigaussian = BiGaussian(
                100_000, sigma_dt=bunch_lengths / 4, seed=1234
            )

            ring = Ring(
                circumference,
            )

            ring.add_elements(
                [profile, cavity, beam_control, lattice],
            )

            simulation = Simulation(
                ring,
                cycle,
            )

            simulation.prepare_beam(beam, bigaussian)

            _dt_tmp = beam._dt.array_local
            _dE_tmp = beam._dE.array_local
            _flags_tmp = beam._flags.array_local
            _ids_tmp = beam._ids.array_local

            for i in range(1, n_bunches):
                beam._dt = DistributedArray(
                    np.append(beam._dt.array_local, _dt_tmp + 10 * t_rf * i)
                )
                beam._dE = DistributedArray(
                    np.append(beam._dE.array_local, _dE_tmp)
                )
                beam._flags = DistributedArray(
                    np.append(beam._flags.array_local, _flags_tmp)
                )
                beam._ids = DistributedArray(
                    np.append(beam._ids.array_local, _ids_tmp)
                )

            beam._dt.array_local += (
                bucket_shift * t_rf + injection_phase_error / 360 * t_rf
            )

            simulation.finalize(
                (beam,),
                n_turns,
            )

            cls.v_ant = np.zeros((n_turns, h // 10), dtype=complex)
            cls.i_beam = np.zeros((n_turns, h // 10), dtype=complex)
            cls.rf_power = np.zeros((n_turns, h // 10), dtype=complex)
            cls.rf_beam_current_phase = np.zeros((n_turns, n_bunches))
            cls.beam_loop_phase = np.zeros(n_turns)

            for i in range(n_turns):
                simulation.turn_i.value = i

                for element in ring.elements.elements:
                    element.track(beam)

                cls.v_ant[i, :] = cavity_control.V_ANT_COARSE[-h // 10 :]
                cls.i_beam[i, :] = cavity_control.I_BEAM_COARSE[-h // 10 :]
                cls.rf_power[i, :] = cavity_control.generator_power()[
                    -h // 10 :
                ]
                cls.beam_loop_phase[i] = beam_control.phi_beam * 180 / np.pi
                cls.rf_beam_current_phase[i, :] = -np.angle(
                    cavity_control.I_BEAM_COARSE[
                        cavity_control.n_coarse
                        + bucket_shift // 10 : cavity_control.n_coarse
                        + bucket_shift // 10
                        + n_bunches
                    ]
                )

            cls.rf_beam_current_phase = np.mean(
                np.unwrap(cls.rf_beam_current_phase) * 180 / np.pi, axis=1
            )
            cls.rf_beam_current_phase = (
                cls.rf_beam_current_phase
                - cls.rf_beam_current_phase[0]
                + injection_phase_error
            )
            cls.beam_loop_phase = (
                cls.beam_loop_phase
                - cls.beam_loop_phase[0]
                + injection_phase_error
            )

        def setup_blond2():
            from blond.legacy.blond2.beam.beam import Beam, Proton
            from blond.legacy.blond2.beam.distributions import (
                bigaussian,
            )
            from blond.legacy.blond2.beam.profile import CutOptions, Profile
            from blond.legacy.blond2.input_parameters.rf_parameters import (
                RFStation,
            )
            from blond.legacy.blond2.input_parameters.ring import Ring
            from blond.legacy.blond2.llrf.beam_feedback import BeamFeedback
            from blond.legacy.blond2.llrf.cavity_feedback import (
                LHCCavityLoop,
                LHCCavityLoopCommissioning,
            )
            from blond.legacy.blond2.trackers.tracker import RingAndRFTracker
            from blond.legacy.blond2.utils import bmath as bm

            bm.use_numba()

            # Options
            PLT_SIMS = False
            DISABLE_PL = False
            ring = Ring(
                circumference, alpha, momentum, Proton(), n_turns=n_turns + 1
            )

            rfstation = RFStation(ring, [h], [voltage], [dphi], n_rf=1)

            # Beam object for the batch
            N_m = n_macroparticles * number_of_bunches
            N_p = bunch_intensity * number_of_bunches
            beam = Beam(ring, N_m, N_p)

            # First generate a single gaussian bunch
            single_bunch = Beam(ring, n_macroparticles, bunch_intensity)
            bigaussian(
                ring,
                rfstation,
                single_bunch,
                sigma_dt=tau_bunch / 4,
                seed=1234,
            )

            # Copy the bunch throughout the batch
            for i in range(number_of_bunches):
                beam.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = (
                    single_bunch.dE
                )
                beam.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
                    single_bunch.dt + i * bunch_spacing * rfstation.t_rf[0, 0]
                )

            # Add final corrections to the bunch positions
            beam.dt += (
                bucket_shift * rfstation.t_rf[0, 0]
                + injection_phase_error * rfstation.t_rf[0, 0] / 360
            )
            beam.dE += injection_energy_error

            # The beam profile
            cut_options = CutOptions(
                cut_left=(-5.5 + bucket_shift) * rfstation.t_rf[0, 0],
                cut_right=(
                    6.5 + number_of_bunches * bunch_spacing + bucket_shift
                )
                * rfstation.t_rf[
                    0,
                    0,
                ],
                n_slices=(10 * number_of_bunches + 12) * 2**5,
            )
            profile = Profile(beam, cut_options)

            # Plot profile
            if PLT_SIMS:
                profile.track()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(profile.bin_centers * 1e6, profile.n_macroparticles)
                ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
                ax.set_ylabel(r"$\lambda (\Delta t)$ [arb. units]")
                ax.set_yticks([])

                plt.show()

            commissioning = LHCCavityLoopCommissioning(
                G_a=G_a,
                G_d=G_d,
                tau_d=tau_d,
                tau_a=tau_a,
                alpha=a_comb,
                G_o=G_otfb,
                open_tuner=True,
                open_rffb=False,
                enable_klystron=False,
            )

            cavity_loop = LHCCavityLoop(
                rfstation,
                profile,
                RFFB=commissioning,
                f_c=rfstation.omega_rf[0, 0] / (2 * np.pi) + delta_f,
                Q_L=Q_L,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                n_pretrack=200,
                n_cavities=8,
                n_h=0,
            )

            # Beam-phase loop
            # Beam Loops
            PL_gain = 1 / (5 * ring.t_rev[0]) * int(not DISABLE_PL)
            SL_gain = PL_gain / 10
            bl_config = {
                "machine": "LHC",
                "PL_gain": PL_gain,
                "SL_gain": SL_gain,
            }

            beam_loop = BeamFeedback(
                ring,
                rfstation,
                profile,
                bl_config,
                CavityFeedback=cavity_loop,
                current_thres=0.5,
            )

            # The RF tracker
            rftracker = RingAndRFTracker(
                rfstation,
                beam,
                Profile=profile,
                interpolation=True,
                BeamFeedback=beam_loop,
                CavityFeedback=cavity_loop,
            )

            # Initialize data arrays
            cls.rf_power_blond2 = np.zeros(
                (n_turns, cavity_loop.n_coarse), dtype=complex
            )
            cls.rf_voltage_blond2 = np.zeros(
                (n_turns, cavity_loop.n_coarse), dtype=complex
            )
            cls.rf_beam_current_blond2 = np.zeros(
                (n_turns, cavity_loop.n_coarse), dtype=complex
            )
            cls.rf_beam_current_phase_blond2 = np.zeros(
                (n_turns, number_of_bunches)
            )
            cls.beam_loop_phase_blond2 = np.zeros(n_turns)

            print(profile.bin_size * 1e12)

            # if DISABLE_PL:
            #    profile.track()
            #    beam_loop.track()
            # Tracking
            profile.track()
            cls.line_density_blond2 = np.copy(profile.n_macroparticles)
            cls.bin_centers_blond2 = np.copy(profile.bin_centers)

            for i in tqdm(range(n_turns)):
                profile.track()
                rftracker.track()
                cavity_loop.generator_power()

                if i == 0:
                    cls.rf_beam_current_fine_blond2 = cavity_loop.I_BEAM_FINE[
                        -profile.n_slices :
                    ]

                cls.rf_power_blond2[i, :] = cavity_loop.generator_power()[
                    -cavity_loop.n_coarse :
                ]
                cls.rf_voltage_blond2[i, :] = cavity_loop.V_ANT_COARSE[
                    -cavity_loop.n_coarse :
                ]
                cls.rf_beam_current_blond2[i, :] = cavity_loop.I_BEAM_COARSE[
                    -cavity_loop.n_coarse :
                ]
                cls.beam_loop_phase_blond2[i] = (
                    beam_loop.phi_beam * 180 / np.pi
                )
                cls.rf_beam_current_phase_blond2[i, :] = -np.angle(
                    cavity_loop.I_BEAM_COARSE[
                        cavity_loop.n_coarse
                        + bucket_shift // 10 : cavity_loop.n_coarse
                        + bucket_shift // 10
                        + number_of_bunches
                    ]
                )

            cls.rf_beam_current_phase_blond2 = np.mean(
                np.unwrap(cls.rf_beam_current_phase_blond2) * 180 / np.pi,
                axis=1,
            )
            cls.rf_beam_current_phase_blond2 = (
                cls.rf_beam_current_phase_blond2
                - cls.rf_beam_current_phase_blond2[0]
                + injection_phase_error
            )
            cls.beam_loop_phase_blond2 = (
                cls.beam_loop_phase_blond2
                - cls.beam_loop_phase_blond2[0]
                + injection_phase_error
            )

            if PLT_SIMS:
                plt.figure("Phase evolution")
                plt.plot(
                    cls.rf_beam_current_phase_blond2,
                    color="black",
                    label="RF beam current",
                )
                plt.plot(
                    cls.beam_loop_phase_blond2,
                    color="r",
                    label="Beam-phase loop",
                )
                plt.legend()
                plt.tight_layout()
                plt.grid()
                plt.xlim(0, n_turns - 1)

                plt.figure("Phase difference")
                plt.plot(
                    100
                    * (
                        cls.rf_beam_current_phase_blond2
                        - cls.beam_loop_phase_blond2
                    )
                    / cls.beam_loop_phase_blond2
                )
                plt.tight_layout()
                plt.grid()
                plt.xlim(0, n_turns - 1)

                plt.show()

        setup_blond3()
        setup_blond2()

    def test_beam_phase_loop(self):
        np.testing.assert_allclose(
            self.beam_loop_phase + 10,
            self.beam_loop_phase_blond2 + 10,
            rtol=4e-5,
            err_msg="Error in phase loop error signal",
        )

    def test_rf_beam_current(self):
        np.testing.assert_allclose(
            self.rf_beam_current_phase + 10,
            self.rf_beam_current_phase_blond2 + 10,
            rtol=4e-5,
            err_msg="Error in turn-by-turn phase of rf beam current",
        )

        np.testing.assert_allclose(
            np.abs(self.i_beam),
            np.abs(self.rf_beam_current_blond2),
            rtol=1e-5,
            err_msg="Error in absolute value of rf beam current",
        )

        np.testing.assert_allclose(
            np.angle(self.i_beam, deg=True),
            np.angle(self.rf_beam_current_blond2, deg=True),
            rtol=2e-3,
            err_msg="Error in phase value of rf beam current",
        )

    def test_rf_voltage_transient(self):
        np.testing.assert_allclose(
            np.abs(self.v_ant),
            np.abs(self.rf_voltage_blond2),
            rtol=9e-6,
            err_msg="Error in absolute value of rf voltage",
        )

        np.testing.assert_allclose(
            np.angle(self.v_ant, deg=True) + 10,
            np.angle(self.rf_voltage_blond2, deg=True) + 10,
            rtol=4e-5,
            err_msg="Error in phase value of rf voltage",
        )

    def test_rf_power_transient(self):
        np.testing.assert_allclose(
            np.abs(self.rf_power),
            np.abs(self.rf_power_blond2),
            rtol=2e-3,
            err_msg="Error in absolute value of rf power",
        )

        np.testing.assert_allclose(
            np.angle(self.rf_power, deg=True),
            np.angle(self.rf_power_blond2, deg=True),
            atol=1e-9,
            err_msg="Error in phase value of rf power",
        )
