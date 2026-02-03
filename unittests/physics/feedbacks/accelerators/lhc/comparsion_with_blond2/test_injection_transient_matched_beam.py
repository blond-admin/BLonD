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


class TestInjectionMatchedBeam(unittest.TestCase):
    blond2_data = np.load(
        Path(
            callers_relative_path(
                "../resources/lhc_cavity_control_injection_power_no_bpl.npz",
                stacklevel=1,
            )
        )
    )

    @classmethod
    def setUpClass(cls):
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

        lattice = DriftSimple(
            orbit_length=circumference, momentum_compaction_factor=alpha
        )

        cavity = MultiHarmonicRfStation(
            voltage=np.array([voltage]),
            phi_rf=np.array([0.0]),
            harmonic=np.array([h]),
            n_harmonics=1,
            main_harmonic_idx=0,
        )

        f_rf = cavity.get_main_harmonic_omega_rf_design(
            rel_beta, lattice.orbit_length
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
            n_pretrack=100,
            Q_L=Q_L,
            tau_loop=tau_loop,
            tau_otfb=tau_comp,
            G_gen=G_gen,
            RFFB=commissioning,
        )

        cavity.attach_cavity_feedback(cavity_control)

        bigaussian = BiGaussian(
            n_macroparticles_per_bunch, sigma_dt=bunch_lengths / 4, seed=1234
        )

        ring = Ring(
            circumference,
        )

        ring.add_elements(
            [profile, cavity, lattice],
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

        for i in range(n_bunches):
            beam._dt[
                i * n_macroparticles_per_bunch : (i + 1)
                * n_macroparticles_per_bunch
            ] = _dt_tmp + 10 * t_rf * i + 1000 * t_rf
            beam._dE[
                i * n_macroparticles_per_bunch : (i + 1)
                * n_macroparticles_per_bunch
            ] = _dE_tmp
            beam._flags[
                i * n_macroparticles_per_bunch : (i + 1)
                * n_macroparticles_per_bunch
            ] = _flags_tmp
            beam._ids[
                i * n_macroparticles_per_bunch : (i + 1)
                * n_macroparticles_per_bunch
            ] = _ids_tmp

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
            cls.v_ant_fine[i, :] = cavity_control.V_ANT_FINE[-profile.n_bins :]
            cls.i_beam[i, :] = cavity_control.I_BEAM_COARSE[-h // 10 :]
            cls.rf_power[i, :] = cavity_control.generator_power()[-h // 10 :]

    def test_line_density(self):
        np.testing.assert_allclose(
            self.line_density,
            self.blond2_data["line_density"],
            rtol=1e-2,
            err_msg="Error in line density",
        )

    def test_beam_current(self):
        # Real part
        np.testing.assert_allclose(
            self.i_beam.real,
            self.blond2_data["i_beam"].real,
            atol=1e-6,
            err_msg="Error in real part of beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.i_beam.imag,
            self.blond2_data["i_beam"].imag,
            atol=1e-6,
            err_msg="Error in imaginary part of beam current",
        )

    def test_gap_voltage(self):
        np.testing.assert_allclose(
            self.v_ant.real,
            self.blond2_data["rf_voltage"].real,
            atol=1e-2,
            err_msg="Error in real part of gap voltage",
        )
        np.testing.assert_allclose(
            self.v_ant.imag,
            self.blond2_data["rf_voltage"].imag,
            atol=1e-2,
            err_msg="Error in imaginary part of gap voltage",
        )

    def test_generator_power_demand(self):
        np.testing.assert_allclose(
            self.rf_power.real,
            self.blond2_data["rf_power"].real,
            atol=1e-9,
            err_msg="Error in real part of rf power",
        )
        np.testing.assert_allclose(
            self.rf_power.imag,
            self.blond2_data["rf_power"].imag,
            atol=1e-9,
            err_msg="Error in imaginary part of rf power",
        )
