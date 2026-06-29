# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover

"""
Induced voltage of a Gaussian bunch in a single resonator: MuSiC.

Compares three ways of obtaining the resonator induced voltage of a single
Gaussian bunch:

* **MuSiC** (:class:`~blond.physics.impedances.music_algorithm.Music`) — the
  exact O(n) time-domain algorithm, evaluated per macro-particle;
* **analytical** — the closed-form Gaussian-bunch-resonator formula
  (independent of BLonD), the ground truth;
* optionally a profile-based **WakeField** with an FFT/convolution solver.

The defaults reproduce the legacy ``EX_11`` resonator (100 MHz, Q=1), which
sits in the *fast-wake* regime ``omega_R * sigma ~ 19``: the wake oscillates
~3 times across the bunch, so the per-particle MuSiC voltage has a low
signal-to-noise ratio and only the *binned mean* tracks the analytical curve.

Things to investigate by editing the constants below:

* ``FREQUENCY_R`` / ``SIGMA_DT`` set ``omega_R * sigma``. Lower ``FREQUENCY_R``
  towards 1e6 (``omega_R * sigma << 1``): the wake barely oscillates across the
  bunch, the per-particle voltage is smooth, and MuSiC matches the analytical
  curve to ~0.1 %. Raise it and the per-particle scatter explodes (noise scales
  with ``omega_R`` while the net signal shrinks), needing more particles /
  coarser bins.
* ``n_macroparticles`` drives the shot noise (``~1/sqrt(N)``).

Notes
-----
MuSiC only supports the ``python`` and ``cpp`` backends; this script selects
one of them automatically.

Authors: Simon Lauber
"""

import numpy as np
from matplotlib import pyplot as plt

from blond import (
    AllowPlotting,
    Beam,
    ConstantMagneticCycle,
    DriftSimple,
    Music,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    TimeDomainFftSolver,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.legacy.blond2.impedances.induced_voltage_analytical import (
    analytical_gaussian_resonator,
)
from blond.testing import pytest_active

# --- parameters to play with (defaults match the legacy EX_11) --------------
R_S = 1e7  # resonator shunt impedance [Ohm]
FREQUENCY_R = 1e8  # resonator frequency [Hz]; lower towards 1e6 for a slow
#                    wake (omega_R*sigma << 1) where MuSiC matches to ~0.1%
Q = 1.0  # quality factor [1]
SIGMA_DT = 3e-8  # RMS bunch length [s]
INTENSITY = 1e12  # beam intensity (real particles) [1]
N_BINS = 40  # bins for the averaged MuSiC-vs-analytical comparison
COMPARE_WAKEFIELD = True  # also overlay a profile-based WakeField FFT solver
# ----------------------------------------------------------------------------


def _ensure_music_backend() -> None:  # pragma: no cover
    """Select a MuSiC-capable backend (cpp if available, else python)."""
    if backend.specials_mode not in ("cpp", "cpp_single_core", "python"):
        try:
            setup_backend("cpp")
        except (FileNotFoundError, OSError):
            setup_backend("python")


if not pytest_active():  # pragma: no cover
    _ensure_music_backend()


def _gaussian_beam(n_macroparticles: int) -> Beam:
    # Gaussian in dt centred at 0, with dE = 0 so the drift stays inert.
    return Beam.simple_gaussian(
        n_macroparticles=n_macroparticles,
        intensity=INTENSITY,
        particle_type=proton,
        dt_scale=SIGMA_DT,
        dE_scale=0.0,
        seed=1000,
    )


def _new_ring() -> Ring:
    # Minimal ring (zero-voltage RF + drift) so a beam can be tracked.
    ring = Ring(circumference=2 * np.pi * 100)
    ring.add_elements(
        (
            SingleHarmonicRFStation(harmonic=1, voltage=0.0, phi_rf=0.0),
            DriftSimple(
                orbit_length=2 * np.pi * 100,
                momentum_compaction_factor=momentum_compaction_factor(
                    transition_gamma=20.0
                ),
            ),
        )
    )
    return ring


def _new_simulation(ring: Ring) -> Simulation:
    return Simulation(
        ring=ring,
        magnetic_cycle=ConstantMagneticCycle(
            reference_particle=proton, value=25.92e9
        ),
    )


def music_induced_voltage(
    n_macroparticles: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Track a Gaussian bunch for one turn; dE == 0 keeps the drift inert, so
    # MuSiC sees the pristine Gaussian. Returns sorted (dt, induced_voltage).
    beam = _gaussian_beam(n_macroparticles)
    ring = _new_ring()
    music = Music(source=Resonators(R_S, FREQUENCY_R, Q))
    ring.add_elements((music,))
    _new_simulation(ring).run_simulation(beams=(beam,), n_turns=1)
    return np.asarray(beam.read_partial_dt()), np.asarray(
        music.induced_voltage
    )


def wakefield_induced_voltage(
    n_macroparticles: int,
) -> tuple[np.ndarray, np.ndarray]:
    # Same bunch through a profile-based WakeField with an FFT solver.
    beam = _gaussian_beam(n_macroparticles)
    ring = _new_ring()
    profile = StaticProfile(-4 * SIGMA_DT, 4 * SIGMA_DT, 400)
    wakefield = WakeField(
        sources=(Resonators(R_S, FREQUENCY_R, Q),),
        solver=TimeDomainFftSolver(),
        profile=profile,
    )
    ring.add_elements((wakefield,))
    _new_simulation(ring).run_simulation(beams=(beam,), n_turns=1)
    return np.asarray(profile.hist_x), np.asarray(wakefield.induced_voltage)


def analytical_induced_voltage(tau: np.ndarray) -> np.ndarray:
    return analytical_gaussian_resonator(
        SIGMA_DT, Q, R_S, 2 * np.pi * FREQUENCY_R, tau, INTENSITY
    ).real


def bin_average(
    dt: np.ndarray, values: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    # Mean of `values` per dt-bin. Averaging beats the per-particle shot noise
    # down by sqrt(particles-per-bin); sparsely-populated tail bins (where it
    # isn't beaten down enough to be meaningful) are masked to NaN.
    edges = np.linspace(-3 * SIGMA_DT, 3 * SIGMA_DT, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_of = np.digitize(dt, edges) - 1
    in_range = (bin_of >= 0) & (bin_of < n_bins)  # drop out-of-window tails
    idx = bin_of[in_range]
    sums = np.bincount(idx, weights=values[in_range], minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    binned = sums / np.maximum(counts, 1)
    binned[counts < 0.02 * counts.max()] = np.nan
    return centers, binned


def max_relative_error(centers: np.ndarray, binned: np.ndarray) -> float:
    analytical = analytical_induced_voltage(centers)
    ok = ~np.isnan(binned)
    return float(
        np.max(np.abs(binned[ok] - analytical[ok]))
        / np.max(np.abs(analytical))
    )


def plot(
    dt: np.ndarray,
    music: np.ndarray,
    centers: np.ndarray,
    binned: np.ndarray,
    wakefield: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    fine = np.linspace(-4 * SIGMA_DT, 4 * SIGMA_DT, 2000)
    step = max(1, len(dt) // 4000)  # subsample the noisy per-particle scatter
    with AllowPlotting():
        plt.figure("EX_29 MuSiC vs analytical")
        plt.plot(
            fine * 1e9,
            analytical_induced_voltage(fine),
            "k-",
            lw=2,
            label="analytical",
        )
        plt.plot(
            dt[::step] * 1e9, music[::step], ".", label="MuSiC (per particle)"
        )
        plt.plot(centers * 1e9, binned, "o-", label="MuSiC (binned mean)")
        if wakefield is not None:
            plt.plot(wakefield[0] * 1e9, wakefield[1], label="WakeField (FFT)")
        plt.xlabel("dt [ns]")
        plt.ylabel("induced voltage [V]")
        plt.title(f"omega_R*sigma = {2 * np.pi * FREQUENCY_R * SIGMA_DT:.2g}")
        plt.legend(loc="upper right")
        plt.tight_layout()


def main():
    _ensure_music_backend()
    n_macroparticles = 5_000 if pytest_active() else 5_000_000
    print(f"omega_R * sigma = {2 * np.pi * FREQUENCY_R * SIGMA_DT:.3g}")

    dt, music = music_induced_voltage(n_macroparticles)
    centers, binned = bin_average(dt, music, N_BINS)
    print(
        "binned MuSiC vs analytical: max rel. error = "
        f"{max_relative_error(centers, binned):.3%}"
    )

    wakefield = (
        wakefield_induced_voltage(n_macroparticles)
        if COMPARE_WAKEFIELD
        else None
    )
    plot(dt, music, centers, binned, wakefield)


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
