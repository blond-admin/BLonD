import unittest

import matplotlib.pyplot as plt
import numpy as np

DEBUG_PLOTTING = False


class TestInjectionMatchedBeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def setup_blond2():
            from blond.legacy.blond2.beam.beam import Beam, Proton
            from blond.legacy.blond2.beam.distributions import (
                bigaussian,
            )
            from blond.legacy.blond2.beam.profile import (
                CutOptions,
                Profile,
            )
            from blond.legacy.blond2.input_parameters.rf_parameters import (
                RFStation,
            )
            from blond.legacy.blond2.input_parameters.ring import Ring
            from blond.legacy.blond2.llrf.cavity_feedback import (
                LHCCavityLoop,
                LHCCavityLoopCommissioning,
            )
            from blond.legacy.blond2.trackers.tracker import (
                FullRingAndRF,
                RingAndRFTracker,
            )

            voltages_tot = 7.9e6
            intensities = 2.3e11
            bunch_lengths = 1.25e-9

            # Constants
            n_macroparticles = int(1e6)  # Macro-particles
            n_bunches = 36  # Number of bunches
            C = 26658.8832  # Machine circumference [m]
            p_s = 450e9  # Synchronous momentum [eV/c]
            h = 35640  # Harmonic number
            dphi = 0  # Phase modulation/offset
            R_over_Q = 45  # Cavity R/Q [Ohms]
            gamma_t = 53.8  # Transition gamma
            alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

            n_turns = 100

            G_a = 6.79e-6  # Analog FB gain [A/V]
            G_d = 10  # Digital FB gain [-]
            tau_loop = 650e-9  # Overall loop delay [s]
            tau_a = 170e-6  # Analog FB delay [s]
            tau_d = 400e-6  # Digital FB delay [s]
            a_comb = 15 / 16  # Comb filter alpha [-]
            Q_L = 20000  # Loaded Quality factor [-]
            G_otfb = 10
            tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
            G_gen = 1
            tau_o = 110e-6

            ring = Ring(C, alpha, p_s, Particle=Proton(), n_turns=n_turns)
            rf = RFStation(
                ring, [h], [voltages_tot], [dphi]
            )  # Assume filamented with SPS emittance
            bunch = Beam(ring, n_macroparticles, intensities)
            bigaussian(ring, rf, bunch, sigma_dt=bunch_lengths / 4, seed=1234)

            beam = Beam(
                ring, n_macroparticles * n_bunches, intensities * n_bunches
            )
            buckets = rf.t_rf[0, 0] * 10

            for i in range(n_bunches):
                beam.dt[i * n_macroparticles : (i + 1) * n_macroparticles] = (
                    bunch.dt[0:n_macroparticles]
                    + 100 * buckets
                    + 10 * rf.t_rf[0, 0] * i
                )
                beam.dE[i * n_macroparticles : (i + 1) * n_macroparticles] = (
                    bunch.dE[0:n_macroparticles]
                )

            profile = Profile(
                beam,
                CutOptions(
                    n_slices=int(2**6 * (10 * n_bunches + 10)),
                    cut_left=(1000 - 5) * rf.t_rf[0, 0],
                    cut_right=(1000 + 10 * n_bunches + 5) * rf.t_rf[0, 0],
                ),
            )

            profile.track()

            RFFB = LHCCavityLoopCommissioning(
                G_a=G_a,
                G_d=G_d,
                tau_d=tau_d,
                tau_a=tau_a,
                alpha=a_comb,
                tau_o=tau_o,
                open_otfb=False,
                G_o=G_otfb,
                mu=-20,
                open_tuner=True,
                d_phi_ad=0,
            )

            CL = LHCCavityLoop(
                rf_station=rf,
                profile=profile,
                f_c=rf.omega_rf[0, 0] / (2 * np.pi) - 5e3,
                I_gen_offset=0,
                n_cavities=8,
                n_pretrack=100,
                Q_L=Q_L,
                R_over_Q=R_over_Q,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                G_gen=G_gen,
                RFFB=RFFB,
            )

            tracker = RingAndRFTracker(
                rf, beam, Profile=profile, CavityFeedback=CL
            )
            tracker = FullRingAndRF([tracker])

            cls.rf_power_blond2 = np.zeros(
                (n_turns, CL.n_coarse), dtype=complex
            )
            cls.i_beam_blond2 = np.zeros((n_turns, CL.n_coarse), dtype=complex)
            cls.v_ant_blond2 = np.zeros((n_turns, CL.n_coarse), dtype=complex)
            cls.line_density_blond2 = np.zeros((n_turns, profile.n_slices))

            for i in range(n_turns):
                profile.track()
                tracker.track()
                cls.rf_power_blond2[i, :] = CL.generator_power()[
                    -CL.n_coarse :
                ]
                cls.i_beam_blond2[i, :] = CL.I_BEAM_COARSE[-CL.n_coarse :]
                cls.v_ant_blond2[i, :] = CL.V_ANT_COARSE[-CL.n_coarse :]
                cls.line_density_blond2[i, :] = profile.n_macroparticles

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
            from blond.experimental.physics.feedbacks.accelerators.lhc.cavity_feedback import (
                LHCCavityLoop,
                LHCCavityLoopCommissioning,
            )
            from blond.generals.distributed.distributed_array import (
                DistributedArray,
            )

            backend.change_backend(Numpy64Bit)
            backend.set_specials("cpp")

            circumference = 26658.8832  # [m]
            momentum = 450e9
            voltage = 7.9e6
            h = 35640
            gamma_t = 53.8
            n_macroparticles_per_bunch = 1_000_000

            n_bunches = 36
            intensity = 2.3e11 * n_bunches
            n_turns = 100
            bunch_lengths = 1.25e-9

            G_a = 6.79e-6  # Analog FB gain [A/V]
            G_d = 10  # Digital FB gain [-]
            tau_loop = 650e-9  # Overall loop delay [s]
            tau_a = 170e-6  # Analog FB delay [s]
            tau_d = 400e-6  # Digital FB delay [s]
            a_comb = 15 / 16  # Comb filter alpha [-]
            Q_L = 20000  # Loaded Quality factor [-]
            G_otfb = 10
            tau_comp = 1200e-9  # Complimentary delay in OTFB [s]
            G_gen = 1
            tau_o = 110e-6
            alpha = 1.0 / gamma_t / gamma_t  # First order mom. comp. factor

            energy = np.sqrt(momentum**2 + proton.mass**2)
            rel_gamma = energy / proton.mass
            rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

            beam = Beam(
                intensity,
                proton,
            )

            cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

            drift = DriftSimple(
                orbit_length=circumference, momentum_compaction_factor=alpha
            )

            cavity = MultiHarmonicRFStation(
                voltage=np.array([voltage]),
                phi_rf=np.array([0.0]),
                harmonic=np.array([h]),
                n_harmonics=1,
                main_harmonic_idx=0,
            )

            f_rf = cavity.get_main_harmonic_omega_rf_design(
                rel_beta, drift.orbit_length
            ) / (2 * np.pi)
            t_rf = 1 / f_rf

            profile = StaticProfile(
                cut_left=(1000 - 5) / f_rf,
                cut_right=(1000 + n_bunches * 10 + 5) / f_rf,
                n_bins=2**6 * (10 + n_bunches * 10),
            )

            # LHC cavity feedback
            commissioning = LHCCavityLoopCommissioning(
                G_a=G_a,
                G_d=G_d,
                tau_d=tau_d,
                tau_a=tau_a,
                alpha=a_comb,
                tau_o=tau_o,
                open_otfb=False,
                G_o=G_otfb,
                mu=-20,
                open_tuner=True,
                d_phi_ad=0,
            )
            cavity_control = LHCCavityLoop(
                profile=profile,
                f_c=f_rf - 5e3,
                n_pretrack=200,
                Q_L=Q_L,
                tau_loop=tau_loop,
                tau_otfb=tau_comp,
                G_gen=G_gen,
                RFFB=commissioning,
                n_cavities=8,
            )

            cavity.attach_cavity_feedback(cavity_control)

            bigaussian = BiGaussian(
                n_macroparticles_per_bunch,
                sigma_dt=bunch_lengths / 4,
                seed=1234,
            )

            ring = Ring(
                circumference,
            )

            ring.add_elements(
                [profile, cavity, drift],
            )

            simulation = Simulation(
                ring,
                cycle,
            )

            simulation.prepare_beam(beam, bigaussian)

            _dt_tmp = beam._dt
            _dE_tmp = beam._dE
            _flags_tmp = beam._flags
            _ids_tmp = beam._ids

            beam._dt = DistributedArray(
                np.zeros(n_bunches * _dt_tmp.local_size)
            )
            beam._dE = DistributedArray(
                np.zeros(n_bunches * _dE_tmp.local_size)
            )
            beam._flags = DistributedArray(
                np.zeros(
                    n_bunches * _flags_tmp.local_size,
                    dtype=_flags_tmp.array_local.dtype,
                )
            )
            beam._ids = DistributedArray(
                np.zeros(
                    n_bunches * _ids_tmp.local_size,
                    dtype=_ids_tmp.array_local.dtype,
                )
            )

            for i in range(n_bunches):
                beam._dt.array_local[
                    i * n_macroparticles_per_bunch : (i + 1)
                    * n_macroparticles_per_bunch
                ] = _dt_tmp.array_local + 10 * t_rf * i + 1000 * t_rf
                beam._dE.array_local[
                    i * n_macroparticles_per_bunch : (i + 1)
                    * n_macroparticles_per_bunch
                ] = _dE_tmp.array_local
                beam._flags.array_local[
                    i * n_macroparticles_per_bunch : (i + 1)
                    * n_macroparticles_per_bunch
                ] = _flags_tmp.array_local
                beam._ids.array_local[
                    i * n_macroparticles_per_bunch : (i + 1)
                    * n_macroparticles_per_bunch
                ] = _ids_tmp.array_local

            simulation.finalize(
                (beam,),
                n_turns,
            )
            cls.line_density = np.zeros((n_turns, profile.n_bins))
            cls.v_ant = np.zeros((n_turns, h // 10), dtype=complex)
            cls.v_ant_fine = np.zeros((n_turns, profile.n_bins), dtype=complex)
            cls.i_beam = np.zeros((n_turns, h // 10), dtype=complex)
            cls.rf_power = np.zeros((n_turns, h // 10), dtype=complex)

            for i in range(n_turns):
                simulation.turn_i.value = i

                for element in ring.elements.elements:
                    element.track(beam)

                cls.line_density[i, :] = profile.hist_y
                cls.v_ant[i, :] = cavity_control.V_ANT_COARSE[-h // 10 :]
                cls.v_ant_fine[i, :] = cavity_control.V_ANT_FINE[
                    -profile.n_bins :
                ]
                cls.i_beam[i, :] = cavity_control.I_BEAM_COARSE[-h // 10 :]
                cls.rf_power[i, :] = cavity_control.generator_power()[
                    -h // 10 :
                ]

        setup_blond2()
        setup_blond3()

    def test_line_density(self):
        np.testing.assert_allclose(
            self.line_density,
            self.line_density_blond2,
            rtol=1e-2,
            err_msg="Error in line density",
        )

    def test_beam_current(self):
        # Real part
        if DEBUG_PLOTTING:
            end_bin = 500
            for _ in range(5):
                plt.plot(self.i_beam.real[_][0:end_bin])
                plt.plot(self.i_beam_blond2.real[_][0:end_bin], ls="--")
            plt.show()
        if DEBUG_PLOTTING:
            end_bin = 500
            for _ in range(5):
                plt.plot(self.i_beam.imag[_][0:end_bin])
                plt.plot(self.i_beam_blond2.imag[_][0:end_bin], ls="--")
            plt.show()
        np.testing.assert_allclose(
            self.i_beam.real,
            self.i_beam_blond2.real,
            atol=1e-6,
            err_msg="Error in real part of beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.i_beam.imag,
            self.i_beam_blond2.imag,
            atol=1e-6,
            err_msg="Error in imaginary part of beam current",
        )

    def test_gap_voltage(self):
        np.testing.assert_allclose(
            self.v_ant.real,
            self.v_ant_blond2.real,
            atol=1e-2,
            err_msg="Error in real part of gap voltage",
        )
        np.testing.assert_allclose(
            self.v_ant.imag,
            self.v_ant_blond2.imag,
            atol=1e-2,
            err_msg="Error in imaginary part of gap voltage",
        )

    def test_generator_power_demand(self):
        np.testing.assert_allclose(
            self.rf_power.real,
            self.rf_power_blond2.real,
            rtol=8e-6,
            err_msg="Error in real part of rf power",
        )
        np.testing.assert_allclose(
            self.rf_power.imag,
            self.rf_power_blond2.imag,
            atol=1e-9,
            err_msg="Error in imaginary part of rf power",
        )
