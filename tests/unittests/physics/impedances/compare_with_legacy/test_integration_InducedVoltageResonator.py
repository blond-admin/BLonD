import unittest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
from blond.physics.impedances.solvers import (
    SingleTurnResonatorConvolutionSolver,
)
from blond.physics.impedances.sources import Resonators
from blond.testing.backend_testing import BLonDTestCase
from blond.testing.helpers import save_blond2_reference_file

from .test_integration_InducedVoltageFreq import (
    Q_factor,
    R_shunt,
    f_res,
)

DEV_PLOT = False

# BLonD 2 only runs to rewrite the reference file, see resources/README.md.
REWRITE_BLOND2_REFERENCE_FILE = False


def _run_blond2(n_macroparticles, n_slices, bunch_length):
    """Run BLonD 2 and return the arrays stored in the reference file."""
    from blond.legacy.blond2.beam.beam import Beam, Proton
    from blond.legacy.blond2.beam.distributions import bigaussian
    from blond.legacy.blond2.beam.profile import CutOptions, Profile
    from blond.legacy.blond2.impedances.impedance import (
        InducedVoltageFreq,
        InducedVoltageResonator,
        InducedVoltageTime,
        TotalInducedVoltage,
    )
    from blond.legacy.blond2.impedances.impedance_sources import Resonators
    from blond.legacy.blond2.input_parameters.rf_parameters import (
        RFStation,
    )
    from blond.legacy.blond2.input_parameters.ring import Ring

    induced_voltage = []

    for solver in (
        InducedVoltageTime,
        InducedVoltageResonator,
        # InducedVoltageFreq,
    ):
        ring = Ring(6911.56, 0.00192, 25.92e9, Proton(), 10)
        rf_station = RFStation(ring, [4620], [0.9e6], [0.0], 1)
        full_beam = Beam(ring, n_macroparticles, 1e10)
        bigaussian(ring, rf_station, full_beam, bunch_length, seed=1)
        # Keeps the BLonD 2 reference file small: beams above 1e5 particles are thinned
        # to about 1e5 (every 10th particle for the 1e6 case: 1.6 MB instead
        # of 16 MB). BLonD 2 computes the induced voltage from this thinned
        # beam, and BLonD 3 loads the same particles from the BLonD 2 reference file,
        # so both still see identical input.
        stride = max(1, n_macroparticles // 100_000)
        beam = Beam(
            ring,
            len(full_beam.dt[::stride]),
            1e10,
            dt=full_beam.dt[::stride].copy(),
            dE=full_beam.dE[::stride].copy(),
        )

        cut_options = CutOptions(
            cut_left=0,
            cut_right=2 * np.pi,
            n_slices=n_slices,
            rf_station=rf_station,
            cuts_unit="rad",
        )
        profile = Profile(beam, cut_options)

        profile.track()
        # R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
        resonator = Resonators(R_shunt, f_res, Q_factor)

        if solver == InducedVoltageTime:
            ind_volt = InducedVoltageTime(
                beam,
                profile,
                [resonator],
            )
        elif solver == InducedVoltageFreq:
            ind_volt = InducedVoltageFreq(beam, profile, [resonator], 1e5)
        elif solver == InducedVoltageResonator:
            ind_volt = InducedVoltageResonator(
                beam,
                profile,
                resonator,
            )
        else:
            raise Exception
        tot_vol = TotalInducedVoltage(beam, profile, [ind_volt])

        tot_vol.induced_voltage_sum()
        induced_voltage.append(tot_vol.induced_voltage)

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
    return dict(
        dt=beam.dt,
        dE=beam.dE,
        cut_left=np.asarray(profile.cut_left),
        cut_right=np.asarray(profile.cut_right),
        n_slices=np.asarray(profile.n_slices),
        induced_voltage=np.asarray(induced_voltage),
    )


def load_blond2(n_macroparticles, n_slices, bunch_length):
    """BLonD 2 results, loaded from the reference file."""
    blond2_reference_path = callers_relative_path(
        "resources/induced_voltage_resonator"
        f"_{n_macroparticles}_{n_slices}_{bunch_length:.3e}_blond2.npz",
        stacklevel=1,
    )
    if REWRITE_BLOND2_REFERENCE_FILE:
        save_blond2_reference_file(
            blond2_reference_path,
            **_run_blond2(n_macroparticles, n_slices, bunch_length),
        )
    with np.load(blond2_reference_path) as blond2_reference:
        return SimpleNamespace(
            dt=blond2_reference["dt"],
            dE=blond2_reference["dE"],
            profile=SimpleNamespace(
                cut_left=blond2_reference["cut_left"].item(),
                cut_right=blond2_reference["cut_right"].item(),
                n_slices=blond2_reference["n_slices"].item(),
            ),
            induced_voltage=blond2_reference["induced_voltage"],
        )


class Blond3:
    def __init__(
        self, n_macroparticles=int(1e6), n_slices=256, bunch_length=1e-9 / 4
    ):
        blond2 = load_blond2(
            n_macroparticles=n_macroparticles,
            n_slices=n_slices,
            bunch_length=bunch_length,
        )
        self.blond2 = blond2

        ring = Ring(circumference=6911.56)
        profile = StaticProfile(
            blond2.profile.cut_left,
            blond2.profile.cut_right,
            blond2.profile.n_slices,
        )
        cavity1 = SingleHarmonicRFStation()
        cavity1.voltage = 0.9e6
        cavity1.phi_rf_design = 0
        cavity1.harmonic = 4620
        drift = DriftSimple(orbit_length=ring.circumference)
        drift.momentum_compaction_factor = momentum_compaction_factor(
            1 / (1 / np.sqrt(0.00192)) ** 2
        )
        # R_shunt, f_res, Q_factor = 5e5, 1e9, 10e10
        resonators = Resonators(
            shunt_impedances=R_shunt,
            center_frequencies=f_res,
            quality_factors=Q_factor,
        )

        beam = Beam(intensity=1e10, particle_type=proton)
        beam.setup_beam(dt=blond2.dt, dE=blond2.dE)
        profile.track(beam)

        wake = WakeField(
            sources=(resonators,),
            solver=SingleTurnResonatorConvolutionSolver(),
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
            try:
                plt.figure(2)
                plt.plot(wake.solver._freq_y)
            except AttributeError:
                pass
        self.induced_voltage = induced_voltage


class TestBothBlonds(BLonDTestCase):
    def setUp(self):
        backend.change_backend(Numpy64Bit)

    def close_in_norm(self, arr_a, arr_b, rtol=1e-12, atol=1e-8):
        norm = np.inf
        a = np.asarray(arr_a)
        b = np.asarray(arr_b)
        diff = a - b
        den = max(
            np.linalg.norm(a, norm),
            np.linalg.norm(b, norm),
            atol / rtol if rtol > 0 else 1.0,
        )
        return np.linalg.norm(diff, norm) <= rtol * den + atol * np.sqrt(
            diff.size
        )

    @pytest.mark.backend_mutation
    def test_integration(self):
        n_macroparticles = int(1e6)
        n_slices = 1024  # this falls apart at low n_slices as the fftconvolve becomes inaccurate (BLonD2)
        bunch_length = 1e-9 / 8
        self.blond3 = Blond3(n_macroparticles, n_slices, bunch_length)

        DEBUG_PLOT = False
        if DEBUG_PLOT:
            plt.title(f"{n_macroparticles} {n_slices} {bunch_length}")
            plt.plot(
                self.blond3.blond2.induced_voltage[0],
                label="blond2 ind_volt time",
            )
            plt.plot(
                self.blond3.blond2.induced_voltage[1],
                label="blond2 ind volt res",
                ls=":",
            )
            # plt.plot(self.blond3.blond2.induced_voltage[2], label="blond2 ind volt freq", ls="--")
            plt.plot(self.blond3.induced_voltage, label="blond3", ls="dashdot")
            plt.legend()
            plt.show()

        for blond2_ind_volt in self.blond3.blond2.induced_voltage:
            try:
                assert self.close_in_norm(
                    blond2_ind_volt, self.blond3.induced_voltage
                )
            except AssertionError:
                np.testing.assert_allclose(
                    blond2_ind_volt,
                    self.blond3.induced_voltage,
                    atol=20,  # of 120000
                )

    @pytest.mark.backend_mutation
    def test_diff_params(self):
        DEBUG_MODE = False
        if DEBUG_MODE:
            n_macroparts = [int(1e4), int(1e5), int(1e6)]
            bunch_lengths = [
                1e-8 / 12,
                1e-9 / 8,
                1e-9 / 4,
            ]
            n_slices_lst = [1024]
        else:
            n_macroparts = [int(2e4)]
            bunch_lengths = [5e-10]
            n_slices_lst = [128]
        for mac_ind, n_macroparticles in enumerate(n_macroparts):
            for slic_ind, n_slices in enumerate(n_slices_lst):
                # for slic_ind, n_slices in enumerate([1024]):
                # for b_ind, bunch_length in enumerate([1e-9 / 4, 1e-9, 4e-9]):
                for b_ind, bunch_length in enumerate(bunch_lengths):
                    self.blond3 = Blond3(
                        n_macroparticles, n_slices, bunch_length
                    )

                    DEBUG_PLOT = False
                    if DEBUG_PLOT:
                        plt.title(
                            f"{n_macroparticles} {n_slices} {bunch_length}"
                        )
                        plt.plot(
                            self.blond3.blond2.induced_voltage[0],
                            label="blond2 ind_volt time",
                        )
                        plt.plot(
                            self.blond3.blond2.induced_voltage[1],
                            label="blond2 ind volt res",
                            ls=":",
                        )
                        # plt.plot(self.blond3.blond2.induced_voltage[2], label="blond2 ind volt res", ls="--")
                        plt.plot(
                            self.blond3.induced_voltage,
                            label="blond3",
                            ls="dashdot",
                        )
                        plt.legend()
                        plt.show()

                    for blond2_ind_volt in self.blond3.blond2.induced_voltage:
                        try:
                            assert self.close_in_norm(
                                blond2_ind_volt, self.blond3.induced_voltage
                            )
                        except AssertionError:
                            np.testing.assert_allclose(
                                blond2_ind_volt,
                                self.blond3.induced_voltage,
                                rtol=1e-2,
                                atol=200,
                            )
