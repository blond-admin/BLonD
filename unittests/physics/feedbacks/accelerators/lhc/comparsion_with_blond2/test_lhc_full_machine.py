import unittest
from pathlib import Path

import numpy as np

from blond import (
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    MultiHarmonicRfStation,
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
from blond.handle_results.helpers import callers_relative_path


class TestLHCFullMachine(unittest.TestCase):
    blond2_data = np.load(
        Path(
            callers_relative_path(
                "../resources/lhc_rf_power_full_machine.npz",
                stacklevel=1,
            )
        )
    )

    @classmethod
    def setUpClass(cls):
        backend.change_backend(Numpy64Bit)
        backend.set_specials("cpp")

        n_bunches = 2748

        circumference = 26658.8832  # [m]
        momentum = 450e9
        intensity = 2.3e11 * n_bunches
        n_macroparticles_per_bunch = 50000
        n_turns = 20
        h = 35640
        gamma_t = 53.8

        n_detuning = 50

        voltages_tot = 7.9e6
        bunch_lengths = 1.25e-9

        # Constants
        R_over_Q = 45  # Cavity R/Q [Ohms]
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
        df_hd = -10.373079819809341e3

        energy = np.sqrt(momentum**2 + proton.mass**2)
        rel_gamma = energy / proton.mass
        rel_beta = np.sqrt(1 - 1 / rel_gamma**2)

        beam = Beam(
            intensity,
            proton,
        )

        cycle = ConstantMagneticCycle(proton, momentum, in_unit="momentum")

        lattice = DriftSimple(
            orbit_length=circumference,
            momentum_compaction_factor=1.0 / gamma_t / gamma_t,
        )

        cavity = MultiHarmonicRfStation(
            voltage=np.array([voltages_tot]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([h]),
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        f_rf = cavity.get_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
        ) / (2 * np.pi)
        f_rev = f_rf / h
        t_rev = 1 / f_rev

        profile = StaticProfile(
            cut_left=0,
            cut_right=t_rev,
            n_bins=int(2**6 * h),
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
            open_tuner=False,
            d_phi_ad=0,
        )
        cavity_control = LHCCavityLoop(
            profile=profile,
            f_c=f_rf + df_hd,
            I_gen_offset=0,
            n_cavities=8,
            n_pretrack=200,
            Q_L=Q_L,
            R_over_Q=R_over_Q,
            tau_loop=tau_loop,
            tau_otfb=tau_comp,
            G_gen=G_gen,
            RFFB=commissioning,
        )
        cavity_control.disable_fine_grid = True
        cavity.attach_cavity_feedback(cavity_control)

        bigaussian = BiGaussian(
            n_macroparticles_per_bunch, sigma_dt=bunch_lengths / 4
        )

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, lattice, cavity],
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

        beam._dt = np.zeros(n_bunches * len(_dt_tmp))
        beam._dE = np.zeros(n_bunches * len(_dE_tmp))
        beam._flags = np.zeros(
            n_bunches * len(_flags_tmp), dtype=_flags_tmp.dtype
        )
        beam._ids = np.zeros(n_bunches * len(_ids_tmp), dtype=_ids_tmp.dtype)

        simulation.finalize(
            (beam,),
            n_turns,
        )

        profile._hist_x = cls.blond2_data["profile_bin_centers"]
        profile._hist_y = cls.blond2_data["profile_n_macroparticles"]

        cls.detunings = np.zeros(n_detuning)

        for i in range(n_detuning):
            cavity_control.track(beam)
            cls.detunings[i] = cavity_control.detuning

        cls.rf_power = cavity_control.generator_power()[
            -cavity_control.n_coarse :
        ]
        cls.rf_power = (
            cls.rf_power
            * np.exp(1j * np.angle(cavity_control.I_GEN_COARSE))[
                -cavity_control.n_coarse :
            ]
        )

        cls.rf_voltage = cavity_control.V_ANT_COARSE[
            -cavity_control.n_coarse :
        ]
        cls.set_point = cavity_control.V_SET[-cavity_control.n_coarse :]
        cls.rf_beam_current = cavity_control.I_BEAM_COARSE[
            -cavity_control.n_coarse :
        ]
        cls.rf_beam_current_fine = cavity_control.I_BEAM_FINE[
            -profile.n_bins :
        ]

    def test_tuner_algorithm(self):
        np.testing.assert_allclose(
            self.detunings,
            self.blond2_data["detunings"],
            atol=1e-8,
            err_msg="Error in tuner algorithm",
        )

    def test_rf_power(self):
        # Real part
        np.testing.assert_allclose(
            self.rf_power.real,
            self.blond2_data["rf_power"].real,
            rtol=6e-7,
            err_msg="Error in real part of rf power",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_power.imag,
            self.blond2_data["rf_power"].imag,
            rtol=5e-5,
            err_msg="Error in imaginary part of rf power",
        )

    def test_rf_voltage(self):
        # Real part
        np.testing.assert_allclose(
            self.rf_voltage.real,
            self.blond2_data["rf_voltage"].real,
            rtol=6e-3,
            err_msg="Error in real part of rf voltage",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_voltage.imag,
            self.blond2_data["rf_voltage"].imag,
            rtol=7e-3,
            err_msg="Error in imaginary part of rf voltage",
        )

    def test_rf_beam_current_coarse(self):
        # Real part
        np.testing.assert_allclose(
            self.rf_beam_current.real,
            self.blond2_data["rf_beam_current"].real,
            atol=1e-8,
            err_msg="Error in real part of coarse-grid rf beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_beam_current.imag,
            self.blond2_data["rf_beam_current"].imag,
            atol=1e-8,
            err_msg="Error in imaginary part of coarse-grid rf beam current",
        )

    def test_rf_beam_current_fine(self):
        # Real part
        np.testing.assert_allclose(
            self.rf_beam_current_fine.real,
            self.blond2_data["rf_beam_current_fine"].real,
            atol=1e-8,
            err_msg="Error in real part of fine-grid rf beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_beam_current_fine.imag,
            self.blond2_data["rf_beam_current_fine"].imag,
            atol=1e-8,
            err_msg="Error in imaginary part of fine-grid rf beam current",
        )

    def test_set_point_voltage(self):
        # Real part
        np.testing.assert_allclose(
            self.set_point.real,
            self.blond2_data["set_point"].real,
            atol=1e-8,
            err_msg="Error in real part of set point voltage",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.set_point.imag,
            self.blond2_data["set_point"].imag,
            atol=1e-8,
            err_msg="Error in imaginary part of set point voltage",
        )
