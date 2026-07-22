"""All wakefield solvers must agree on the voltage of an inductive impedance.

Regression test: for odd FFT lengths, `PeriodicFreqSolver` returned one
sample too few (`irfft` without `n=` assumes an even signal length), so
the induced voltage did not match `profile.hist_x`. Additionally, the
derivative kernel of `InductiveImpedance` reconstructed the time-domain
bin width from `freq_x` under the same even-length assumption, giving
slightly wrong voltages in `PeriodicFreqSolver` and `TimeDomainFftSolver`.

Authors: Simon Lauber
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from blond import (
    ConstantMagneticCycle,
    InductiveImpedance,
    Ring,
    Simulation,
    StaticProfile,
    WakeField,
    copy_to_cpu,
    proton,
)
from blond.core.beam.beams import ProbeBeam
from blond.physics.impedances.solvers import (
    InductiveImpedanceSolver,
    PeriodicFreqSolver,
    TimeDomainFftSolver,
)

_DEV_DRAW = os.getenv("DEV_DRAW", "False").lower() == "true"


class _FakeProfile(StaticProfile):
    """Static profile with a hand-written histogram."""

    def __init__(self, cut_left: float, cut_right: float, profile_y):
        super().__init__(
            cut_left=cut_left, cut_right=cut_right, n_bins=len(profile_y)
        )
        self._hist_y = np.array(profile_y, dtype=float)
        self.hist_y_to_density_factor = 1.0


def _calc_induced_voltage(solver, profile_y) -> np.ndarray:
    profile = _FakeProfile(cut_left=-0.5, cut_right=0.5, profile_y=profile_y)
    circumference = profile.cut_right - profile.cut_left
    wakefield = WakeField(
        sources=(InductiveImpedance(123),),
        solver=solver,
        profile=profile,
    )
    beam = ProbeBeam(
        particle_type=proton,
        intensity=1e12,
        reference_total_energy=1e12,
        dt=np.array([0]),
    )
    ring = Ring(circumference=circumference)
    ring.add_elements([profile, wakefield])
    Simulation(
        ring=ring,
        magnetic_cycle=ConstantMagneticCycle(
            reference_particle=beam.particle_type,
            value=beam.reference.total_energy,
        ),
    )
    wakefield.calc_induced_voltage(beam=beam)
    return copy_to_cpu(wakefield.induced_voltage)


# n_zeros=5 -> n_bins=15: odd FFT length in `PeriodicFreqSolver`
# n_zeros=4 -> n_bins=13: odd FFT length in `TimeDomainFftSolver`
#              (next_fast_len(2 * 13) = 27)
@pytest.mark.integration
@pytest.mark.parametrize("n_zeros", [4, 5])
def test_solvers_agree_on_inductive_impedance(n_zeros):
    pad = n_zeros * [0]
    profile_y = pad + [1, 2, 3, 2, 1] + pad

    t_span = 1.0  # cut_right - cut_left in _calc_induced_voltage
    reference = _calc_induced_voltage(InductiveImpedanceSolver(), profile_y)
    assert reference.shape == (len(profile_y),)

    results = {
        "PeriodicFreqSolver": _calc_induced_voltage(
            PeriodicFreqSolver(t_periodicity=t_span), profile_y
        ),
        "TimeDomainFftSolver": _calc_induced_voltage(
            TimeDomainFftSolver(), profile_y
        ),
    }

    if _DEV_DRAW:
        plt.plot(reference, label="InductiveImpedanceSolver")
        i = 0
        for name, induced_voltage in results.items():
            plt.plot(induced_voltage, ("-", "--", "..")[i], label=name)
            i += 1
        plt.legend()
        plt.show()

    for name, induced_voltage in results.items():
        assert induced_voltage.shape == reference.shape, name
        np.testing.assert_allclose(
            induced_voltage,
            reference,
            rtol=1e-10,
            atol=np.abs(reference).max() * 1e-12,
            err_msg=name,
        )
