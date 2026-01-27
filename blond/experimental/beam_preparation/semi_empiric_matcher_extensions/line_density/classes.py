from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from blond.experimental.beam_preparation.semi_empiric_matcher_extensions.line_density.callables import (
    state_vector_to_hammilton_coordinates,
    state_vector_to_histogram,
)

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray


class ProfileMatcher:
    def __init__(
        self,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
    ):
        self.hist_x = hist_x
        self.hist_y = hist_y
        self.recenter = False
        self.animate_fitting = True
        self.plot_result = True
        self.plot_result_blocking = True
        self.maxiter = 100

        self.atol = 1e-6

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

        density = self._solve_for_density(
            hamilton_2D=hamilton_2D,
            histogram_desired=hist_y_interp,
            animate_fitting=self.animate_fitting,
        )

        if self.plot_result:
            self._plot_result(
                time_grid=time_grid,
                deltaE_grid=deltaE_grid,
                hamilton_2D=hamilton_2D,
                density=density,
                hist_x_interp=hist_x_interp,
                hist_y_interp=hist_y_interp,
            )
            if self.plot_result_blocking:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.1)

        return density

    def _solve_for_density(
        self,
        hamilton_2D: NumpyArray,
        histogram_desired: NumpyArray,
        animate_fitting: bool = False,
    ) -> NumpyArray:
        histogram_desired = histogram_desired.copy()
        histogram_desired_normalized = (
            histogram_desired / histogram_desired.sum()
        )

        state_vector = histogram_desired.copy()  # initial guess
        histogram = state_vector_to_histogram(
            state_vector=state_vector,
            hamilton_2D=hamilton_2D,
        )
        update_state_vector = histogram_desired.sum() / histogram.sum()
        state_vector *= update_state_vector

        histogram = state_vector_to_histogram(
            state_vector=state_vector,
            hamilton_2D=hamilton_2D,
        )
        histogram_normalized = histogram / histogram.sum()

        previous_histogram_normalized = histogram_normalized
        for i in tqdm(range(self.maxiter)):
            update_state_vector = (1 + histogram_desired) / (1 + histogram)
            state_vector *= update_state_vector

            histogram = state_vector_to_histogram(
                state_vector=state_vector,
                hamilton_2D=hamilton_2D,
            )
            histogram_normalized = histogram / histogram.sum()
            if (
                histogram_normalized - previous_histogram_normalized
            ).max() < self.atol:
                break
            previous_histogram_normalized = histogram_normalized

            if animate_fitting:
                plt.figure(0)
                plt.clf()
                plt.title(f"Iteration {i}")
                plt.plot(histogram_desired_normalized)
                plt.plot(histogram_normalized, "--")
                plt.xlabel("Time [s]")
                plt.ylabel("Density [arb. unit]")
                plt.draw()
                plt.pause(0.1)
        density = state_vector_to_hammilton_coordinates(
            state_vector=state_vector,
            hamilton_2D=hamilton_2D,
        )
        # normalize
        density = density / np.sum(density)

        return density

    @staticmethod
    def _plot_result(
        time_grid,
        deltaE_grid,
        hamilton_2D,
        density,
        hist_x_interp,
        hist_y_interp,
    ):
        plt.figure()
        ax1 = plt.subplot(3, 1, 1)
        plt.title("Hamilton")
        plt.pcolor(
            time_grid, deltaE_grid, hamilton_2D, cmap="viridis", shading="auto"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [eV]")

        ax2 = plt.subplot(3, 1, 2, sharex=ax1, sharey=ax1)
        plt.title("Density Distribution")

        plt.pcolor(
            time_grid,
            deltaE_grid,
            density,
            cmap="viridis",
            shading="auto",
            vmin=1e-24,
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Energy [eV]")

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        plt.title("Line Density (normalized)")
        plt.plot(
            hist_x_interp, hist_y_interp / hist_y_interp.sum(), label="Target"
        )
        plt.plot(
            hist_x_interp,
            density.sum(axis=1) / density.sum(),
            linestyle="--",
            label="Fitted",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Density [arb. unit]")

        plt.legend()
