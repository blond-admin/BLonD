import unittest

import numpy as np

from blond3 import (
    Ring,
    Beam,
    SingleHarmonicCavity,
    DriftSimple,
    proton,
    StaticProfile,
    Simulation,
    ConstantMagneticCycle, MultiHarmonicCavity,
)
from blond3.physics.feedbacks.accelerators.lhc.cavity_feedback import (
    LHCCavityLoopCommissioning,
    LHCCavityLoop,
)


class TestLHCOpenDrive(unittest.TestCase):
    def setUp(self):
        # Bunch parameters (dummy)
        N_b = 1e9  # Intensity
        N_p = 50000  # Macro-particles
        # Machine and RF parameters
        C = 26658.883  # Machine circumference [m]
        p_s = 450e9  # Synchronous momentum [eV/c]
        h = 35640  # Harmonic number
        V = 4e6  # RF voltage [V]
        dphi = 0  # Phase modulation/offset
        gamma_t = 53.8  # Transition gamma
        alpha = 1 / gamma_t**2  # First order mom. comp. factor

        # Initialise necessary classes
        # ring = Ring(C, alpha, p_s, particle=Proton(), n_turns=1)
        ring = Ring(circumference=C)
        # self.rf = RFStation(ring, [h], [V], [dphi])
        rf = MultiHarmonicCavity(n_harmonics=1, main_harmonic_idx=0)
        rf.harmonic = np.array([h])
        rf.voltage = np.array([V])
        rf.phi_rf = np.array([dphi])
        self.rf = rf
        ring.add_element(rf)
        ring.add_drifts(
            n_drifts_per_section=1,
            n_sections=1,
            driftclass=DriftSimple,
            transition_gamma=gamma_t,
        )
        # beam = Beam(ring, N_p, N_b)
        beam = Beam(
            n_particles=N_b,
            particle_type=proton,
        )
        beam.ratio = N_b / N_p
        # self.profile = Profile(beam)
        self.profile = StaticProfile(0, 1, 8)  # TODO kwargs should matter
        sim = Simulation(
            ring=ring,
            magnetic_cycle=ConstantMagneticCycle(
                reference_particle=proton,
                value=p_s,
                in_unit="momentum",
            ),
        )
        beam.reference_total_energy = sim.magnetic_cycle.get_total_energy_init(
            0, 0, beam.particle_type
        )
        # Test in open loop, on tune
        self.RFFB = LHCCavityLoopCommissioning(open_drive=True)
        omega = self.rf.calc_omega(beam_beta=beam.reference_beta,
                                   ring_circumference=ring.circumference)
        rf._omega_rf = omega # TODO FIXME REMOVE
        self.f_c = float(omega)/(2*np.pi)

    def test_setup(self):
        pass  # see if setUp() works

    def test_1(self):
        CL = LHCCavityLoop(
            self.rf,
            self.profile,
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.2778,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        self.assertAlmostEqual(V_ant, 0.49817991, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1] * 1e-3
        self.assertAlmostEqual(P_gen, 34.7277780000, places=10)

    def test_2(self):
        CL = LHCCavityLoop(
            self.rf,
            self.profile,
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.2778,
            n_cavities=8,
            n_pretrack=0,
            Q_L=60000,
            R_over_Q=45,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        self.assertAlmostEqual(V_ant, 1.26745787, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1] * 1e-3
        self.assertAlmostEqual(P_gen, 104.1833340000, places=10)

    def test_3(self):
        CL = LHCCavityLoop(
            self.rf,
            self.profile,
            f_c=self.f_c,
            G_gen=1,
            I_gen_offset=0.2778,
            n_cavities=8,
            n_pretrack=0,
            Q_L=20000,
            R_over_Q=90,
            tau_loop=650e-9,
            tau_otfb=1472e-9,
            RFFB=self.RFFB,
        )
        CL.track_one_turn()
        # Steady-state antenna voltage [MV]
        V_ant = np.mean(np.absolute(CL.V_ANT_COARSE[-10:])) * 1e-6
        self.assertAlmostEqual(V_ant, 0.99635982, places=7)
        # Updated generator current [A]
        I_gen = np.mean(np.absolute(CL.I_GEN_COARSE[-CL.n_coarse :]))
        self.assertAlmostEqual(I_gen, 0.2778000000, places=10)
        # Generator power [kW]
        P_gen = CL.generator_power()[-1] * 1e-3
        self.assertAlmostEqual(P_gen, 69.4555560000, places=10)
