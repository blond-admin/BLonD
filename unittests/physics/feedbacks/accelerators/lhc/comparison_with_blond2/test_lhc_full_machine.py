"""Compare blond3 and blond2 LHC cavity feedback for the full machine."""

import unittest

import numpy as np
import pytest

from .support import blond2_reference

# The blond3 setup switches the global backend (Numpy64Bit + numba specials).
pytestmark = pytest.mark.backend_mutation

n_bunches = 2748

circumference = 26658.8832  # [m]
momentum = 450e9
intensity = 2.3e11 * n_bunches
n_macroparticles_per_bunch = 50000
n_turns = 20
h = 35640
gamma_t = 53.8
alpha = 1.0 / gamma_t / gamma_t

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

batch_spacings = np.array(
    [
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        330,
        80,
        80,
        80,
        330,
        80,
        80,
        80,
        0,
    ]
)


def _run_blond2() -> dict[str, np.ndarray]:
    """
    Run the frozen blond2 full-machine simulation.

    Only executed when the pinned reference file is absent or regeneration is
    requested (see :func:`support.blond2_reference`); the legacy code and the
    ``seed=1234`` make the outputs deterministic.

    Returns
    -------
    dict
        The blond2 comparison signals plus the full-machine beam histogram
        (``profile_bin_centers_blond2`` / ``profile_n_macroparticles_blond2``)
        that the blond3 setup injects into its own profile.
    """
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.distributions import (
        bigaussian,
    )
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.input_parameters.rf_parameters import (
        RFStation,
    )
    from blond.legacy.blond2.input_parameters.ring import Ring
    from blond.legacy.blond2.llrf.cavity_feedback import (
        LHCCavityLoop,
        LHCCavityLoopCommissioning,
    )

    batch_lengths = np.ones(38, dtype=int) * 72
    batch_lengths = np.concatenate(([12], batch_lengths), dtype=int)

    injection_scheme = np.zeros(np.sum(batch_lengths), dtype=int)
    NB = len(injection_scheme)

    ring = Ring(circumference, alpha, momentum, Particle=Proton(), n_turns=1)
    rf = RFStation(
        ring, [h], [voltages_tot], [0.0]
    )  # Assume filamented with SPS emittance
    bunch = Beam(ring, n_macroparticles_per_bunch, intensity / NB)
    bigaussian(ring, rf, bunch, sigma_dt=bunch_lengths / 4, seed=1234)

    beam = Beam(ring, n_macroparticles_per_bunch * NB, intensity)
    buckets = rf.t_rf[0, 0] * 10

    n_batch = 0
    n_bunch = 0
    db = 0
    for i in range(len(injection_scheme)):
        injection_scheme[i] = db
        n_bunch += 1
        if n_bunch == batch_lengths[n_batch]:
            n_bunch = 0
            db += batch_spacings[n_batch]
            n_batch += 1
        else:
            db += 10

    for i in range(len(injection_scheme)):
        beam.dt[
            i * n_macroparticles_per_bunch : (i + 1)
            * n_macroparticles_per_bunch
        ] = (
            bunch.dt[0:n_macroparticles_per_bunch]
            + 100 * buckets
            + injection_scheme[i] * rf.t_rf[0, 0]
        )
        beam.dE[
            i * n_macroparticles_per_bunch : (i + 1)
            * n_macroparticles_per_bunch
        ] = bunch.dE[0:n_macroparticles_per_bunch]

    profile = Profile(
        beam,
        CutOptions(
            n_slices=int(2**6 * (35640)),
            cut_left=0,  # 80 * buckets,
            cut_right=rf.t_rev[
                0
            ],  # 80 * buckets + tot_buckets * rf.t_rf[0, 0]
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
        open_tuner=False,
        d_phi_ad=0,
        enable_klystron=False,
    )

    CL = LHCCavityLoop(
        rf_station=rf,
        profile=profile,
        f_c=rf.omega_rf[0, 0] / (2 * np.pi) + df_hd,
        I_gen_offset=0,
        n_cavities=8,
        n_pretrack=200,
        Q_L=Q_L,
        R_over_Q=R_over_Q,
        tau_loop=tau_loop,
        tau_otfb=tau_comp,
        G_gen=G_gen,
        RFFB=RFFB,
    )
    CL.disable_fine_grid = True

    detunings = np.zeros(n_detuning)

    for i in range(n_detuning):
        CL.track()
        detunings[i] = CL.detuning

    transient = CL.generator_power()
    transient = transient * np.exp(1j * np.angle(CL.I_GEN_COARSE))

    return {
        "rf_power_blond2": transient[-CL.n_coarse :],
        "rf_voltage_blond2": CL.V_ANT_COARSE[-CL.n_coarse :],
        "rf_beam_current_blond2": CL.I_BEAM_COARSE[-CL.n_coarse :],
        "profile_bin_centers_blond2": profile.bin_centers,
        "profile_n_macroparticles_blond2": profile.n_macroparticles,
        "detunings_blond2": detunings,
        "rf_beam_current_fine_blond2": CL.I_BEAM_FINE[-profile.n_slices :],
        "set_point_blond2": CL.V_SET[-CL.n_coarse :],
    }


class TestLHCFullMachine(unittest.TestCase):
    """
    Compare blond3 and blond2 cavity feedback signals for the full LHC.

    Note that the blond3 side does not histogram its own particles: it
    injects blond2's beam histogram directly into its profile (the dummy
    macro-particles only define the beam's macro-particle count). The
    beam-current comparisons therefore validate the current computation for
    a *given* line density, not the particle-to-histogram step itself --
    that step is cross-checked independently by the ``test_line_density``
    comparisons of the injection-transient test classes.
    """

    @classmethod
    def setUpClass(cls):  # noqa: PLR0915
        """Run the blond3 simulation against the pinned blond2 reference."""

        def setup_blond3():
            """Run the blond3 full-machine simulation and store its results."""
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

            backend.change_backend(Numpy64Bit)
            backend.set_specials("numba")

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

            cavity = MultiHarmonicRFStation(
                voltage=np.array([voltages_tot]),
                phi_rf=np.array([0.0]),
                harmonic=np.array([h]),
                n_harmonics=1,
                main_harmonic_idx=0,
            )

            f_rf = cavity.calc_main_harmonic_omega_rf_design(
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

            # No seed: the generated coordinates are discarded below (the
            # particle arrays are zeroed and the histogram is injected from
            # blond2); only the macro-particle count is used.
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

            beam._dt.array_local = np.zeros(n_bunches * _dt_tmp.local_size)
            beam._dE.array_local = np.zeros(n_bunches * _dE_tmp.local_size)
            beam._flags.array_local = np.zeros(
                n_bunches * _flags_tmp.local_size,
                dtype=_flags_tmp.array_local.dtype,
            )
            beam._ids.array_local = np.zeros(
                n_bunches * _ids_tmp.local_size,
                dtype=_ids_tmp.array_local.dtype,
            )

            simulation.finalize(
                (beam,),
                n_turns,
            )

            profile._hist_x = cls.profile_bin_centers_blond2
            profile._hist_y = cls.profile_n_macroparticles_blond2
            # The histogram is injected directly (bypassing profile.track()),
            # so set the density factor track() would have set. 1/sum(hist_y)
            # normalises to the full beam and matches blond2's 1/n_macro here
            # (the cut spans the whole turn, so no particles are lost).
            profile.hist_y_to_density_factor = 1.0 / np.sum(profile._hist_y)

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

        # The loader must run before `setup_blond3`: blond3 consumes the
        # pinned blond2 histogram (see the class docstring).
        for key, value in blond2_reference(
            "full_machine", _run_blond2
        ).items():
            setattr(cls, key, value)
        setup_blond3()

    def test_tuner_algorithm(self):
        """Check the tuner-algorithm detunings match blond2."""
        np.testing.assert_allclose(
            self.detunings,
            self.detunings_blond2,
            atol=1e-8,
            err_msg="Error in tuner algorithm",
        )

    def test_rf_power(self):
        """Check the real and imaginary RF power match blond2."""
        # Real part
        np.testing.assert_allclose(
            self.rf_power.real,
            self.rf_power_blond2.real,
            rtol=6e-7,
            err_msg="Error in real part of rf power",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_power.imag,
            self.rf_power_blond2.imag,
            rtol=5e-5,
            err_msg="Error in imaginary part of rf power",
        )

    def test_rf_voltage(self):
        """Check the real and imaginary RF voltage match blond2."""
        # Real part
        np.testing.assert_allclose(
            self.rf_voltage.real,
            self.rf_voltage_blond2.real,
            atol=4e-9,
            err_msg="Error in real part of rf voltage",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_voltage.imag,
            self.rf_voltage_blond2.imag,
            atol=7e-3,
            err_msg="Error in imaginary part of rf voltage",
        )

    def test_rf_beam_current_coarse(self):
        """Check the real and imaginary coarse-grid RF beam current match blond2."""
        # Real part
        np.testing.assert_allclose(
            self.rf_beam_current.real,
            self.rf_beam_current_blond2.real,
            atol=1e-8,
            err_msg="Error in real part of coarse-grid rf beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_beam_current.imag,
            self.rf_beam_current_blond2.imag,
            atol=1e-8,
            err_msg="Error in imaginary part of coarse-grid rf beam current",
        )

    def test_rf_beam_current_fine(self):
        """Check the real and imaginary fine-grid RF beam current match blond2."""
        # Real part
        np.testing.assert_allclose(
            self.rf_beam_current_fine.real,
            self.rf_beam_current_fine_blond2.real,
            atol=1e-9,
            err_msg="Error in real part of fine-grid rf beam current",
        )
        # Imaginary part
        np.testing.assert_allclose(
            self.rf_beam_current_fine.imag,
            self.rf_beam_current_fine_blond2.imag,
            atol=1e-9,
            err_msg="Error in imaginary part of fine-grid rf beam current",
        )

    def test_set_point_voltage(self):
        """Check the set-point voltage matches blond2 and stays real."""
        # Real part
        np.testing.assert_allclose(
            self.set_point.real,
            self.set_point_blond2.real,
            atol=1e-9,
            err_msg="Error in real part of set point voltage",
        )
        # Both codes place the set point on the real I/Q axis, so the
        # imaginary parts are identically zero and comparing them against
        # each other would be vacuous. Assert the invariant instead, so a
        # phase-convention change on either side is caught.
        np.testing.assert_array_equal(
            self.set_point.imag,
            0.0,
            err_msg="blond3 set point voltage left the real I/Q axis",
        )
        np.testing.assert_array_equal(
            self.set_point_blond2.imag,
            0.0,
            err_msg="blond2 set point voltage left the real I/Q axis",
        )
