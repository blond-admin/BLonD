# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

from __future__ import annotations

# pragma: no cover
import logging
import math
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import float64, prange, void
from numba.core.types import int32

from blond import (
    Beam,
    BeamObservationOncePerTurn,
    DriftSimple,
    RFStationPhaseObservation,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    proton,
)
from blond.cycles.magnetic_cycle import MagneticCyclePerTurn
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


def get_test_profile():
    # Parameters
    mean = 2.5e-9 / 2  # Mean of the distribution
    std_dev = 2.5e-9 / 8  # Standard deviation
    size = 10000  # Number of data points

    # Generate random data from a Gaussian distribution
    data = np.random.normal(loc=mean, scale=std_dev, size=size)

    # Get the histogram (density=False for raw counts)
    hist_y, bin_edges = np.histogram(data, bins=512, density=False)
    hist_x = bin_edges[0:-1] + np.diff(bin_edges[:2])[0] / 2
    if True:
        hist_x = np.linspace(*(0, 2.5e-9))
        hist_y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(
            -((hist_x - mean) ** 2) / (2 * std_dev**2)
        )
    return hist_x, hist_y


class ProfileMatcher:
    def __init__(
        self,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
    ):
        self.hist_x = hist_x
        self.hist_y = hist_y
        self.recenter = False

    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray | CupyArray,
        deltaE_grid: NumpyArray | CupyArray,
        hamilton_2D: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """Use this function with the `SemiEmpiricMatcher`."""
        if self.recenter:
            mid = time_grid.shape[1] // 2
            center_ham = np.average(
                time_grid[:, mid],
                weights=hamilton_2D[:, mid].max() - hamilton_2D[:, mid],
            )
            center_prof = np.average(self.hist_x, weights=self.hist_y)
            correction = center_ham - center_prof

        else:
            correction = 0.0
        hist_x_interp = time_grid[:, 0]
        hist_y_interp = np.interp(
            hist_x_interp,
            self.hist_x + correction,  # todo if recenter
            self.hist_y,
            left=0,
            right=0,
        )

        density = solve_for_density(hamilton_2D, hist_y_interp)
        density = density / np.sum(density)

        self.plot_result(density, hist_y_interp)

        return density

    @staticmethod
    def plot_result(density, hist_y_interp):
        plt.figure(0)
        plt.clf()
        plt.subplot(1, 3, 1)
        ax = plt.subplot(1, 3, 2)
        ax.matshow(density)
        plt.subplot(1, 3, 3)
        plt.plot(hist_y_interp / hist_y_interp.sum(), label="hist_y_interp")
        plt.plot(density.sum(axis=1) / density.sum(), label="density sum")
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.show()
        plt.matshow(density)
        plt.colorbar()
        plt.show()


def state_vector_to_hammilton_coordinates(state_vector, hamilton_2D):
    import numpy as np
    from tqdm import tqdm

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    density = np.zeros(hamilton_2D.shape, float)
    print("_gen_density start")
    t0 = time.time()
    _gen_density(H_change, density, hamilton_2D, mid, state_vector)
    print(time.time() - t0)
    print("_gen_density stop")
    return density


@numba.njit(
    void(float64[:], float64[:, :], float64[:, :], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_density(H_change, density_write, hamilton_2D, mid, state_vector):
    h_shape_0 = hamilton_2D.shape[0]
    h_shape_1 = hamilton_2D.shape[1]
    n_states = state_vector.shape[0]

    # Preload the mid-column once
    h_mid = hamilton_2D[:, mid]

    # Precompute sigma, sigma², and cutoff windows
    sigma = np.empty(n_states)
    inv_two_sigma_sq = np.empty(n_states)
    e_min = np.empty(n_states)
    e_max = np.empty(n_states)

    # to remove calculation from inner loop
    for i in range(n_states):
        s = 5.0 * H_change[i]
        sigma[i] = s
        inv_two_sigma_sq[i] = -1.0 / (2.0 * s * s)
        e_i = h_mid[i]
        e_min[i] = e_i - 5.0 * s
        e_max[i] = e_i + 5.0 * s

    for idx in prange(h_shape_0 * h_shape_1):
        u = idx % h_shape_1
        v = idx // h_shape_1
        h_u_v = hamilton_2D[u, v]

        acc = 0.0

        for i in range(n_states):
            # Skip if outside cutoff
            if h_u_v < e_min[i] or h_u_v > e_max[i]:
                continue

            dE = h_u_v - h_mid[i]
            w = np.exp(dE * dE * inv_two_sigma_sq[i])
            acc += w * state_vector[i]

        density_write[u, v] = acc


def state_vector_to_histogram(state_vector, hamilton_2D):
    import numpy as np

    mid = hamilton_2D.shape[1] // 2

    # Precompute gradient
    H_1d = hamilton_2D[:, mid]
    H_change = np.abs(np.gradient(H_1d, edge_order=2))

    histogram = np.zeros(hamilton_2D.shape[0], float)
    print("gen_hist start")
    t0 = time.time()
    _gen_hist(H_change, hamilton_2D, histogram, mid, state_vector)
    print("gen_hist stop")
    print(time.time() - t0)
    return histogram


@numba.njit(
    void(float64[:], float64[:, :], float64[:], int32, float64[:]),
    parallel=True,
    fastmath=True,
)
def _gen_hist(H_change, hamilton_2D, histogram_write, mid, state_vector):
    histogram_write[:] = 0.0

    num_states = state_vector.shape[0]
    h_shape_0 = hamilton_2D.shape[0]
    h_shape_1 = hamilton_2D.shape[1]

    # Precompute mid-column energies
    hamilton_mid = hamilton_2D[:, mid]

    # Precompute sigma, cutoff windows, and Gaussian prefactors
    inv_two_sigma2 = np.empty(num_states)
    emin = np.empty(num_states)
    emax = np.empty(num_states)

    for i in range(num_states):
        s = 5.0 * H_change[i]
        inv_two_sigma2[i] = -1.0 / (2.0 * s * s)
        e_i = hamilton_mid[i]
        emin[i] = e_i - 5.0 * s
        emax[i] = e_i + 5.0 * s

    # Main loop
    for u in prange(h_shape_0):
        acc_u = 0.0
        h_u_min = hamilton_2D[u, :].min()

        # Cache row pointer for faster access

        for i in range(num_states):
            e_max_i = emax[i]
            if e_max_i < h_u_min:
                continue
            e_i = hamilton_mid[i]
            s_i = state_vector[i]
            inv2s2 = inv_two_sigma2[i]
            e_min_i = emin[i]

            for v in range(h_shape_1):
                h = hamilton_2D[u, v]

                # Skip expensive exp() if outside cutoff
                if h < e_min_i or h > e_max_i:
                    continue

                dE = h - e_i
                acc_u += np.exp(dE * dE * inv2s2) * s_i

        histogram_write[u] = acc_u


def solve_for_density(hamilton_2D, histogram_desired):
    state_vector = histogram_desired.copy()  # initial guess
    histogram = state_vector_to_histogram(
        state_vector=state_vector,
        hamilton_2D=hamilton_2D,
    )
    update_state_vector = histogram_desired.sum() / histogram.sum()
    state_vector *= update_state_vector

    for i in range(3):
        histogram = state_vector_to_histogram(
            state_vector=state_vector,
            hamilton_2D=hamilton_2D,
        )
        update_state_vector = (1 + histogram_desired) / (1 + histogram)
        state_vector *= update_state_vector

        plt.figure(0)
        plt.clf()
        plt.plot(histogram_desired / histogram_desired.sum())
        plt.plot(histogram / histogram.sum(), "--")
        plt.draw()
        plt.pause(0.1)
    density = state_vector_to_hammilton_coordinates(state_vector, hamilton_2D)

    return density


def main():
    ring = Ring(26658.883)

    rf_station = SingleHarmonicRFStation()
    rf_station.harmonic = 35640
    rf_station.voltage = 6e6
    rf_station.phi_rf = 0

    N_TURNS = int(1e3)

    energy_cycle = MagneticCyclePerTurn(
        value_init=450e9,
        values_after_turn=np.linspace(450e9, 450e9, N_TURNS),
        reference_particle=proton,
    )

    drift1 = DriftSimple(
        orbit_length=26658.883,
    )
    drift1.transition_gamma = 55.759505
    beam1 = Beam(
        intensity=1e9,
        particle_type=proton,
    )

    sim = Simulation.from_locals(locals())
    sim.print_one_turn_execution_order()

    hist_x, hist_y = get_test_profile()
    matcher = ProfileMatcher(hist_x=hist_x, hist_y=hist_y)

    sim.prepare_beam(
        beam=beam1,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=(0, 2.5e-9),
            n_macroparticles=1e6,
            seed=0,
            maxiter_intensity_effects=0,
            hamilton_to_density_function=matcher.hamilton_to_density_function,
            hamilton_to_density_kwargs=dict(),
            animate=True,
        ),
    )

    phase_observation = RFStationPhaseObservation(
        each_turn_i=1,
        rf_station=rf_station,
    )
    bunch_observation = BeamObservationOncePerTurn(each_turn_i=1)

    def custom_action(simulation: Simulation, beam: Beam):  # pragma: no cover
        if simulation.turn_i.value % 10 != 0:
            return

        plt.hist2d(
            beam.read_partial_dt(),
            beam.read_partial_dE(),
            bins=256,
            range=[(0, 2.5e-9), (-4e8, 4e8)],
        )
        plt.xlim((0, 2.5e-9))
        plt.draw()
        plt.pause(0.1)
        plt.clf()

    try:
        sim.load_results(
            beams=(beam1,),
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
        )
        print(
            f"Loaded {phase_observation.common_filepath}"
        )  # pragma: no cover
    except (FileNotFoundError, AssertionError):
        sim.run_simulation(
            beams=(beam1,),
            n_turns=N_TURNS,
            observe=(phase_observation, bunch_observation),
            callbacks=custom_action,
        )
    ANIMATE = False
    if ANIMATE:  # pragma: no cover
        plt.plot(phase_observation.phases)
        plt.figure()
        for i in range(N_TURNS):
            plt.clf()
            plt.hist2d(
                bunch_observation.dts[i, :],
                bunch_observation.dEs[i, :],
                bins=256,
                range=[[0, 2.5e-9], [-4e8, 4e8]],
            )
            plt.draw()
            plt.pause(0.1)

        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
