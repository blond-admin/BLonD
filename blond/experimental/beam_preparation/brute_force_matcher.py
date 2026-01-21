# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/


from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from blond import BoxLosses, Simulation
from blond.beam_preparation.base import MatchingRoutine
from blond.core.beam.base import BeamBaseClass


class BruteForceMatcher(MatchingRoutine):
    """
    Brute-force beam matching routine.

    This matcher initializes the beam using uniformly spaced macroparticles
    within specified time and energy limits and iteratively runs a simulation
    to obtain a matched beam distribution.

    Parameters
    ----------
    time_limit : tuple[float, float]
        Lower and upper limits for the time coordinate, in [s].
        The user should adjust this until they find their matched bunch, they
        can inspect using the animate flag.
    energy_limit : tuple[float, float]
        Lower and upper limits for the energy deviation, in [eV].
        The user should adjust this until they find their matched bunch, they
        can inspect using the animate flag.
    n_macroparticles : int
        Number of macroparticles used to initialize the beam.
    n_iter : int
        Number of simulation iterations to perform.
    animate: bool
        Whether or not to display the simulation animation.
    animate_pause_time: float
        Time to pause the simulation animation, default 0.1, in [s].
    every_iter_to_plot : int, optional
        A snapshot of the beam is
        produced every ``n_iter / every_iter_to_plot`` iterations.
        Default is ``10``.
    purge : bool, optional
        If ``True``, macroparticles outside user-defined phase-space limits
        are removed during the matching process. Default is ``False``.
    purge_limit_time : tuple[float, float], optional
        Lower and upper bounds in time, in [s]. Used to purge particles when
        ``purge=True``. If ``None``, no time-based purging is applied.
    purge_limit_energy : tuple[float, float], optional
        Lower and upper bounds in energy deviation, in [eV]. Used to purge
        particles when ``purge=True``. If ``None``, no energy-based purging
        is applied.
    """

    def __init__(
        self,
        time_limit: tuple[float, float],
        energy_limit: tuple[float, float],
        n_macroparticles: int,
        n_iter: int,
        animate: bool = True,
        animate_pause_time: float = 0.1,
        every_iter_to_plot: int = 10,
        purge: bool = False,
        purge_limit_time: tuple[float, float] = None,
        purge_limit_energy: tuple[float, float] = None,
    ) -> None:
        super().__init__()
        self.time_limit = time_limit
        self.energy_limit = energy_limit
        self.n_macroparticles = n_macroparticles
        self.n_iter = n_iter
        self.animate = animate
        self.animate_pause_time = animate_pause_time
        self.every_iter_to_plot = every_iter_to_plot
        self.purge = purge
        self.purge_limit_time = purge_limit_time
        self.purge_limit_energy = purge_limit_energy

    def prepare_beam(self, simulation: Simulation, beam: BeamBaseClass):
        """
        Prepare and match the beam using a brute-force approach.

        The beam is initialized with uniformly distributed macroparticles
        in time and energy. A copy of the simulation is then run multiple
        times to iteratively evolve the beam towards a matched state.

        Parameters
        ----------
        simulation : Simulation
            Simulation object used to track the beam.
        beam : BeamBaseClass
            Beam instance to be initialized and matched.
        """

        n = int(np.sqrt(self.n_macroparticles))

        dt_vals = np.linspace(self.time_limit[0], self.time_limit[1], n)
        dE_vals = np.linspace(self.energy_limit[0], self.energy_limit[1], n)

        dt_grid, dE_grid = np.meshgrid(dt_vals, dE_vals)

        dt_init = dt_grid.ravel()
        dE_init = dE_grid.ravel()

        beam.setup_beam(dt=dt_init, dE=dE_init)

        if self.animate:
            plt.ion()
            fig, ax = plt.subplots()

            # Beam (updated each frame)

            scat = beam.plot_scatter(ax=ax, s=8, label="Beam", color="C0")

            # Bounding box
            rect = Rectangle(
                (self.time_limit[0], self.energy_limit[0]),
                self.time_limit[1] - self.time_limit[0],
                self.energy_limit[1] - self.energy_limit[0],
                linewidth=2,
                edgecolor="black",
                facecolor="none",
                label="Initial limits",
            )
            ax.add_patch(rect)

            ax.set_xlabel("dt [s]")
            ax.set_ylabel("dE []")
            ax.legend(loc="upper right")
            ax.set_title("Brute-force beam matching")

            step = max(1, self.n_iter // self.every_iter_to_plot)

            ax.set_xlim(self.time_limit[0], self.time_limit[1])
            ax.set_ylim(self.energy_limit[0], self.energy_limit[1])
            ax.set_autoscale_on(False)

            # --------------------------------------------------
            # Matching loop
            # --------------------------------------------------
        for i in range(self.n_iter):
            sim_copy = deepcopy(simulation)
            sim_copy.run_simulation(beams=[beam], n_turns=1)

            if self.animate and (i % step == 0 or i == self.n_iter - 1):
                scat.remove()
                scat = beam.plot_scatter(ax=ax, s=8, label="Beam", color="C0")
                ax.set_title(f"Iteration {i + 1}/{self.n_iter}")

                plt.pause(self.animate_pause_time)

        if self.animate:
            plt.ioff()

        if self.purge:
            BoxLosses(
                t_min=self.purge_limit_time[0],
                t_max=self.purge_limit_time[1],
                e_min=self.purge_limit_energy[0],
                e_max=self.purge_limit_energy[1],
                purge_flagged_macroparticles=True,
            ).track(beam)

            beam.purge_flagged_entries()
