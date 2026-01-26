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
from scipy.sparse import lil_matrix
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

    def hamilton_to_density_function_no(
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
        n_writes = np.zeros_like(hamilton_2D)
        density = np.zeros_like(hamilton_2D)
        matrix = np.zeros((len(hist_y_interp), np.prod(hamilton_2D.shape)))

        for i in range(2, len(hist_y_interp) - 2):
            prev_energy = hamilton_2D[i - 2, mid]
            this_energy = hamilton_2D[i, mid]
            next_energy = hamilton_2D[i + 2, mid]
            e_min = this_energy - 2 * (this_energy - prev_energy)
            e_max = this_energy + 2 * (next_energy - this_energy)
            select = (hamilton_2D <= e_min) & (hamilton_2D >= e_max)
            indices_2d = np.where(select)
            j = np.ravel_multi_index(indices_2d, dims=select.shape)
            matrix[i, j] += 1

        hist_y_residual = hist_y_interp
        for i in range(1):
            print(i)
            density += (hist_y_residual @ matrix).reshape(hamilton_2D.shape)
            # matrix_pinv = np.linalg.pinv(matrix)
            # density = (matrix_pinv @ hist_y_residual).reshape(
            #    hamilton_2D.shape)

            hist_y_result = np.sum(density, axis=1)

            plt.figure(1)
            plt.clf()
            plt.plot(hist_y_result / np.sum(hist_y_result))
            plt.plot(hist_y_interp / np.sum(hist_y_interp), "--")
            hist_y_residual = -(
                hist_y_interp / np.sum(hist_y_interp)
                - hist_y_result / np.sum(hist_y_result)
            )
            plt.plot(hist_y_residual, "--")
            plt.draw()
            plt.pause(0.1)

        return density

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

        matrix_smooth = self.get_smooting_matrix(hamilton_2D.shape)

        matrix1, matrix2 = self.get_transformation_matrix(
            hamilton_2D=hamilton_2D, hist_y_n_bins=len(hist_y_interp)
        )

        # Convert to CSR for fast algebra
        matrix1 = (matrix_smooth @ matrix1).tocsr()
        matrix2 = matrix2.tocsr()

        M = matrix2 @ matrix1
        b = hist_y_interp

        x = np.zeros_like(b)

        relaxation = 0.1
        while True:
            if False:
                # algebraic reconstruction technique
                for i in range(M.shape[0]):
                    a_i = M[i, :]
                    b_i = b[i]

                    # Skip if measurement vector is zero to avoid division by zero
                    a_i_norm_sq = (a_i @ a_i.T)[0, 0]
                    if a_i_norm_sq == 0:
                        continue

                    # Compute residual between observed and predicted measurement
                    residual_ = b_i - a_i @ x

                    # Calculate update scaled by relaxation parameter
                    update = (relaxation * residual_ / a_i_norm_sq) * a_i

                    # Update only masked elements of solution vector
                    x += update

                    # Enforce non-negativity constraint
                    x = np.maximum(x, 0)
            if True:
                from scipy.sparse.linalg import lsqr

                result = lsqr(
                    M,
                    b,
                )

                x = result[0]
                break
        density = (matrix1 @ x).reshape(hamilton_2D.shape)
        density[density < 0] = 0

        plt.figure(0)
        plt.clf()
        ax = plt.subplot(1, 2, 1)
        ax.matshow(density)
        plt.subplot(1, 2, 2)
        plt.plot(hist_y_interp, label="hist_y_interp")
        residual = M @ x
        plt.plot(residual, label="residual")
        plt.legend()
        plt.draw()
        plt.pause(0.1)
        plt.show()

        plt.matshow(density)
        plt.colorbar()
        plt.show()

        return density

    def get_transformation_matrix(self, hamilton_2D, hist_y_n_bins):
        mid = hamilton_2D.shape[1] // 2

        hamilton_2d_shape = hamilton_2D.shape
        matrix1 = lil_matrix(
            (np.prod(hamilton_2d_shape), hist_y_n_bins), dtype=float
        )
        matrix2 = lil_matrix(
            (hist_y_n_bins, np.prod(hamilton_2d_shape)), dtype=float
        )
        H_change = np.abs(np.gradient(hamilton_2D[:, mid], edge_order=2))
        for i in tqdm(
            range(2, hist_y_n_bins - 2),
            desc="preparing matrix",
        ):
            this_energy = hamilton_2D[i, mid]
            for smoothing_i in range(1, 1 + 2):  # create linear fallof
                e_min = this_energy - 5 * smoothing_i * H_change[i]
                e_max = this_energy + 5 * smoothing_i * H_change[i]
                if e_max < e_min:
                    e_min, e_max = e_max, e_min
                if e_max == e_min:
                    raise Exception
                select = (hamilton_2D >= e_min) & (hamilton_2D <= e_max)

                indices_2d = np.where(select)
                j = np.ravel_multi_index(indices_2d, dims=hamilton_2d_shape)
                for jj in j:
                    matrix1[jj, i] += 1

                for jj in range(
                    i * hamilton_2d_shape[1],
                    (i + 1) * hamilton_2d_shape[1],
                ):
                    matrix2[i, jj] += 1
        return matrix1, matrix2

    def get_smooting_matrix(self, hamilton_2D_shape, radius=2, w_center=0.5):
        nx, ny = hamilton_2D_shape
        matrix_smooth = lil_matrix((nx * ny, nx * ny), dtype=float)

        # total weight for neighbors
        w_neighbor_total = 1.0 - w_center

        for i in tqdm(range(nx), desc="preparing smoothing"):
            for j in range(ny):
                row = np.ravel_multi_index((i, j), (nx, ny))

                # center weight
                matrix_smooth[row, row] = w_center

                # neighbors within radius
                neighbors = []
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        if di == 0 and dj == 0:
                            continue  # skip center
                        ni, nj = i + di, j + dj
                        if 0 <= ni < nx and 0 <= nj < ny:
                            neighbors.append((ni, nj))

                if neighbors:
                    w_neighbor = w_neighbor_total / len(neighbors)
                    for ni, nj in neighbors:
                        col = np.ravel_multi_index((ni, nj), (nx, ny))
                        matrix_smooth[row, col] = w_neighbor

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
