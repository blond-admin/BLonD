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
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix, diags, vstack
from scipy.sparse.linalg import lsqr
from tqdm import tqdm

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

# Oversampling factor for potential well calculation
_POTENTIAL_WELL_OVERSAMPLING = 10

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


logging.basicConfig(level=logging.INFO)


def get_profile():
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
    return hist_x + 1, hist_y


class ProfileMatcher:
    def __init__(
        self,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
    ):
        self.hist_x = hist_x
        self.hist_y = hist_y

    def hamilton_to_density_function(
        self,
        time_grid: NumpyArray | CupyArray,
        deltaE_grid: NumpyArray | CupyArray,
        hamilton_2D: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        mid = time_grid.shape[1] // 2
        center_ham = np.average(time_grid[:, mid], weights=hamilton_2D[:, mid])
        center_prof = np.average(self.hist_x, weights=self.hist_y)
        hist_x_interp = time_grid[:, 0]
        hist_y_interp = np.interp(
            hist_x_interp,
            self.hist_x + (center_ham - center_prof),  # todo if recenter
            self.hist_y,
            left=0,
            right=0,
        )

        (
            state_vector_to_hamilton,
            hamilton_to_histogram,
            hamilton_to_smoothness,
        ) = self.get_transformation_matrix(
            hamilton_2D=hamilton_2D, hist_y_n_bins=len(hist_y_interp)
        )
        # matrix_smooth = self.get_smooting_matrix(hamilton_2D.shape)

        # Convert to CSR for fast algebra
        # state_vector_to_hamilton = (matrix_smooth @ state_vector_to_hamilton).tocsr()
        state_vector_to_hamilton = state_vector_to_hamilton.tocsr()
        hamilton_to_histogram = hamilton_to_histogram.tocsr()

        state_vector_to_histogram = (
            hamilton_to_histogram @ state_vector_to_hamilton
        )
        histogram = hist_y_interp

        if False:
            x = self.solve_art()
        if True:
            x = self.solve_lgs(histogram, state_vector_to_histogram)

        plt.figure()
        plt.plot(hamilton_to_smoothness @ state_vector_to_hamilton @ x)
        plt.show()
        density = (state_vector_to_hamilton @ x).reshape(hamilton_2D.shape)
        density[density < 0] = 0

        self.plot_result(density, hist_y_interp, state_vector_to_histogram, x)

        return density

    @staticmethod
    def plot_result(density, hist_y_interp, state_vector_to_histogram, x):
        plt.figure(0)
        plt.clf()
        plt.subplot(1, 3, 1)
        plt.plot(x)
        ax = plt.subplot(1, 3, 2)
        ax.matshow(density)
        plt.subplot(1, 3, 3)
        plt.plot(hist_y_interp, label="hist_y_interp")
        residual = state_vector_to_histogram @ x
        plt.plot(residual, label="residual")
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.show()
        plt.matshow(density)
        plt.colorbar()
        plt.show()

    @staticmethod
    def solve_art():
        x = np.zeros_like(histogram)
        relaxation = 0.01
        weights = np.empty(state_vector_to_histogram.shape[0])
        for i in range(state_vector_to_histogram.shape[0]):
            a_i = state_vector_to_histogram[i, :]
            a_i_norm_sq = (a_i @ a_i.T)[0, 0]
            weights[i] = a_i_norm_sq
        indices = np.arange(state_vector_to_histogram.shape[0])[
            np.argsort(weights)
        ]
        for _ in range(20):
            # algebraic reconstruction technique
            for i in indices:
                a_i = state_vector_to_histogram[i, :]
                b_i = histogram[i]

                # Skip if measurement vector is zero to avoid division by zero
                a_i_norm_sq = (a_i @ a_i.T)[0, 0]
                if a_i_norm_sq == 0:
                    continue

                # Compute residual between observed and predicted measurement
                residual_ = b_i - a_i @ x

                # Calculate update scaled by relaxation parameter
                update = (relaxation * residual_ / a_i_norm_sq) * a_i

                if i == 511 and _ == 0:
                    plt.subplot(4, 1, 1)
                    plt.plot(residual_.flatten(), label="residual_")
                    plt.legend()

                    plt.subplot(4, 1, 2)
                    plt.plot(a_i.toarray().flatten(), "o", label="a_i")
                    plt.legend()

                    plt.subplot(4, 1, 3)
                    plt.plot(update, label="update")
                    plt.legend()

                    plt.subplot(4, 1, 4)
                    plt.plot()
                    plt.show()
                # Update only masked elements of solution vector
                x += update

                # Enforce non-negativity constraint
                x = np.maximum(x, 0)
            # x = gaussian_filter1d(x, sigma=2)  # keep solution smooth
        return x

    @staticmethod
    def solve_lgs(histogram, state_vector_to_histogram):
        x = lsqr(state_vector_to_histogram, histogram)[0]
        return x

    @staticmethod
    def get_transformation_matrix(hamilton_2D, hist_y_n_bins):
        import numpy as np
        from scipy.sparse import coo_matrix
        from tqdm import tqdm

        mid = hamilton_2D.shape[1] // 2
        h_shape = hamilton_2D.shape
        n_elements = np.prod(h_shape)

        # Precompute gradient
        H_1d = hamilton_2D[:, mid]
        H_change = np.abs(np.gradient(H_1d, edge_order=2))

        # Sparse matrix builders
        rows_matrix1, cols_matrix1, data_matrix1 = [], [], []
        rows_matrix2, cols_matrix2 = [], []
        rows_matrix3, cols_matrix3, data_matrix3 = [], [], []

        for i in tqdm(range(2, hist_y_n_bins - 2), desc="preparing matrix"):
            this_energy = hamilton_2D[i, mid]

            sigma = 5 * H_change[i]

            if sigma == 0:
                continue

            # Gaussian weight per (i,j)
            delta_E = hamilton_2D - this_energy
            weights_ij = np.exp(-(delta_E**2) / (2.0 * sigma**2))

            # Optional cutoff window (keeps sparsity under control)
            e_min = this_energy - 10.0 * sigma
            e_max = this_energy + 10.0 * sigma
            mask = (hamilton_2D >= e_min) & (hamilton_2D <= e_max)
            weights_ij *= mask

            # Normalize per i (optional but usually correct)
            s = weights_ij[:, mid].sum()
            if s > 0:
                weights_ij /= s

            # Visualization (debug)
            if False:  # set True to inspect
                plt.figure(10)
                plt.clf()
                plt.matshow(weights_ij.T, fignum=10)
                plt.colorbar()
                plt.title(f"i = {i}")
                plt.pause(0.1)

            # Sparse indices
            indices_2d = np.nonzero(weights_ij)
            flat_indices = np.ravel_multi_index(indices_2d, dims=h_shape)
            vals = weights_ij[indices_2d]

            rows_matrix1.extend(flat_indices)
            cols_matrix1.extend([i] * len(flat_indices))
            data_matrix1.extend(vals)

            # Matrix 2 (row i covers entire row i in Hamiltonian)
            row_indices = np.arange(i * h_shape[1], (i + 1) * h_shape[1])
            rows_matrix2.extend([i] * len(row_indices))
            cols_matrix2.extend(row_indices)

            # Matrix 2 (row i covers entire row i in Hamiltonian)

            i -= 1
            rows_matrix3.append(i)
            cols_matrix3.append(i + h_shape[1] // 2)
            data_matrix3.append(1)
            i += 1
            rows_matrix3.append(i)
            cols_matrix3.append(i + h_shape[1] // 2)
            data_matrix3.append(-2)

            i += 1
            rows_matrix3.append(i)
            cols_matrix3.append(i + h_shape[1] // 2)
            data_matrix3.append(1)

        # Build sparse matrices
        state_vector_to_hamilton = coo_matrix(
            (data_matrix1, (rows_matrix1, cols_matrix1)),
            shape=(n_elements, hist_y_n_bins),
            dtype=float,
        )

        hamilton_to_histogram = coo_matrix(
            (np.ones(len(rows_matrix2)), (rows_matrix2, cols_matrix2)),
            shape=(hist_y_n_bins, n_elements),
            dtype=float,
        )

        hamilton_to_smoothness = coo_matrix(
            (data_matrix3, (rows_matrix3, cols_matrix3)),
            shape=(hist_y_n_bins, n_elements),
            dtype=float,
        )

        return (
            state_vector_to_hamilton,
            hamilton_to_histogram,
            hamilton_to_smoothness,
        )

    @staticmethod
    def get_smooting_matrix(hamilton_2D_shape, radius=2, w_center=0.1):
        nx, ny = hamilton_2D_shape
        N = nx * ny
        w_neighbor_total = 1.0 - w_center

        rows = []
        cols = []
        data = []

        # Precompute neighbor offsets within the radius (excluding the center)
        offsets = [
            (di, dj)
            for di in range(-radius, radius + 1)
            for dj in range(-radius, radius + 1)
            if not (di == 0 and dj == 0)
        ]

        for i in tqdm(range(nx), desc="preparing smoothing"):
            for j in range(ny):
                row = np.ravel_multi_index((i, j), (nx, ny))

                # Add center weight
                rows.append(row)
                cols.append(row)
                data.append(w_center)

                # Valid neighbors
                neighbors = [
                    (i + di, j + dj)
                    for di, dj in offsets
                    if 0 <= i + di < nx and 0 <= j + dj < ny
                ]

                if neighbors:
                    w_neighbor = w_neighbor_total / len(neighbors)
                    for ni, nj in neighbors:
                        col = np.ravel_multi_index((ni, nj), (nx, ny))
                        rows.append(row)
                        cols.append(col)
                        data.append(w_neighbor)

        # Build sparse matrix efficiently
        matrix_smooth = coo_matrix((data, (rows, cols)), shape=(N, N))
        return matrix_smooth.tocsr()


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

    hist_x, hist_y = get_profile()
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
