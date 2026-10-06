import unittest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c, e, m_p

from blond import (
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
)
from blond.core.backends.backend import Numpy64Bit, backend
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.solvers import PeriodicFreqSolver
from blond.physics.impedances.sources import InductiveImpedance
from blond.testing.backend_testing import BLonDTestCase
from blond.testing.helpers import save_blond2_reference_file

DEV_PLOT = False

# BLonD 2 only runs to rewrite the reference file, see resources/README.md.
REWRITE_BLOND2_REFERENCE_FILE = False


def _run_blond2():
    """Run BLonD 2 and return the arrays stored in the reference file."""
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.distributions import bigaussian
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.impedances.impedance import (
        InductiveImpedance,
    )
    from blond.legacy.blond2.input_parameters.rf_parameters import (
        RFStation,
    )
    from blond.legacy.blond2.input_parameters.ring import Ring

    E_0 = m_p * c**2 / e  # [eV]
    ring = Ring(
        2 * np.pi * 25.0,
        1 / 4.4**2,
        np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
        Proton(),
        1,
    )

    rf_station = RFStation(
        ring,
        [1],
        [8e3],
        [np.pi],
        1,
    )

    full_beam = Beam(ring, 10000001, 1e11)
    bucket_length = 2.0 * np.pi / rf_station.omega_rf[0, 0]

    bigaussian(ring, rf_station, full_beam, 180e-9 / 4, seed=1)
    # Keeps the BLonD 2 reference file small (1.6 MB instead of 160 MB): only every
    # 100th bigaussian particle is kept. BLonD 2 computes the induced
    # voltage from this thinned beam, and BLonD 3 loads the same particles
    # from the BLonD 2 reference file, so both still see identical input.
    beam = Beam(
        ring,
        len(full_beam.dt[::100]),
        1e11,
        dt=full_beam.dt[::100].copy(),
        dE=full_beam.dE[::100].copy(),
    )

    number_slices = int(100 * 2.5)

    profile = Profile(
        beam,
        CutOptions(
            cut_left=0, cut_right=bucket_length, n_slices=number_slices
        ),
    )

    inductive_impedance = InductiveImpedance(
        beam, profile, [100] * 1, rf_station
    )
    inductive_impedance.induced_voltage_generation()
    induced_voltage = inductive_impedance.induced_voltage
    if DEV_PLOT:
        plt.figure(0)
        plt.plot(profile.bin_centers, profile.n_macroparticles)
        plt.figure(1)

        plt.plot(induced_voltage, label="blond2")
        plt.legend()
    return dict(
        dt=beam.dt,
        dE=beam.dE,
        cut_left=np.asarray(profile.cut_left),
        cut_right=np.asarray(profile.cut_right),
        n_slices=np.asarray(profile.n_slices),
        induced_voltage=np.asarray(induced_voltage),
    )


def load_blond2():
    """BLonD 2 results, loaded from the reference file."""
    blond2_reference_path = callers_relative_path(
        "resources/inductive_impedance_blond2.npz", stacklevel=1
    )
    if REWRITE_BLOND2_REFERENCE_FILE:
        save_blond2_reference_file(blond2_reference_path, **_run_blond2())
    with np.load(blond2_reference_path) as blond2_reference:
        return SimpleNamespace(
            beam=SimpleNamespace(
                dt=blond2_reference["dt"], dE=blond2_reference["dE"]
            ),
            profile=SimpleNamespace(
                cut_left=blond2_reference["cut_left"].item(),
                cut_right=blond2_reference["cut_right"].item(),
                n_slices=blond2_reference["n_slices"].item(),
            ),
            induced_voltage=blond2_reference["induced_voltage"],
        )


class Blond3:
    def __init__(self):
        blond2 = load_blond2()
        self.blond2 = blond2
        circumference = 2 * np.pi * 25.0
        ring = Ring(circumference=circumference)
        drift = DriftSimple(orbit_length=circumference)
        drift.momentum_compaction_factor = momentum_compaction_factor(4.4)
        cavity = SingleHarmonicRFStation()
        cavity.harmonic = 1
        cavity.voltage = 8e3
        cavity.phi_rf_design = np.pi
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )

        wake = WakeField(
            sources=(InductiveImpedance(100 * 1),),
            # solver=InductiveImpedanceSolver(),
            solver=PeriodicFreqSolver(
                t_periodicity=5.720344547649417e-07, allow_next_fast_len=True
            ),
            profile=profile,
        )
        ring.add_elements((drift, cavity, profile, wake), reorder=True)
        beam = Beam(intensity=1e11, particle_type=proton)

        E_0 = m_p * c**2 / e  # [eV]

        sim = Simulation(
            ring=ring,
            magnetic_cycle=ConstantMagneticCycle(
                value=np.sqrt((E_0 + 1.4e9) ** 2 - E_0**2),
                reference_particle=proton,
            ),
        )
        beam.setup_beam(
            dt=blond2.beam.dt,
            dE=blond2.beam.dE,
            reference_total_energy=sim.magnetic_cycle.get_total_energy_init(
                particle_type=beam.particle_type
            ),
        )
        profile.track(beam)

        self.induced_voltage = wake.calc_induced_voltage(beam)
        if DEV_PLOT:
            plt.figure(0)
            plt.plot(profile.hist_x, profile.hist_y, ".-")
            plt.figure(1)
            plt.plot(self.induced_voltage, "--", color="C1", label="blond3")
            plt.legend()


class TestBothBlonds(BLonDTestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)
        self.blond3 = Blond3()
        if DEV_PLOT:
            plt.show()

    @pytest.mark.backend_mutation
    def test_induced_voltage(self):
        np.testing.assert_allclose(
            self.blond3.blond2.induced_voltage + 1,
            self.blond3.induced_voltage + 1,
            rtol=1e-12,
        )
