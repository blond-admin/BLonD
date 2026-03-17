# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

import numpy as np
import sympy as sp
import tqdm as tqdm
from matplotlib import pyplot as plt
from scipy.constants import c as c0

from blond import DriftSimple, Simulation, SingleHarmonicRFStation
from blond.beam_preparation.base import MatchingRoutine
from blond.core.beam.base import BeamBaseClass
from blond.experimental.cbh_matching.bch_expansion import bch_lattice


class BCHMatcher(MatchingRoutine):
    """
    Uses sympy to create a symbolic representation of the Hamiltonian with Lie algebra.
    $e^{:H_{drift}:}$
    The Hamiltonians are extracted from the ring.
    Then, the Hamiltonian is expanded up to arbitrary order with the CBH expansion.
    From here, the distribution is matched, where f, the distribution, is a function of
    H, the BCH formula expanded Hamiltonian with Lie algebra, f(H).

    For now, does not work with a desired emittance. In progress.
    """

    def __init__(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
        n_macroparticles: int,
        order: int = 1,
        distribution="Gaussian",
        plot: bool = True,
        time_window_limit: tuple = None,
        energy_window_limit: tuple = None,
    ):
        """

        Parameters
        ----------
        simulation
            Simulation instance.
        beam
            Beam instance.
        n_macroparticles
            Number of macroparticles to sample for matching.
        order
            Order of CBH expansion.
        distribution
            Distribution to match f(H).
        plot
            If True, plot the Hamiltonian expansion.
        """
        self.simulation = simulation
        self.beam = beam
        self.H_list = None
        self.n_macroparticles = n_macroparticles
        self.order = order
        self.distribution = distribution
        self.plot = plot
        self.time_window_limit = time_window_limit
        self.energy_window_limit = energy_window_limit

    def prepare_beam(self, simulation: Simulation, beam: BeamBaseClass):
        """
        Sets up the beam via beam.setup_beam with the matched dt, dE coords.

        Parameters
        ----------
        simulation
            Simulation instance.
        beam
            Beam instance.
        """
        self.build_hamiltonian_model()
        self.make_sympy_map()
        self.find_bucket_level()
        self.build_numeric_hamiltonian()

        if self.plot:
            self.plot_hamiltonian(
                q_range=(self.time_window_limit[0], self.time_window_limit[1]),
                p_range=(
                    self.energy_window_limit[0],
                    self.energy_window_limit[1],
                ),
            )

        self.sample_matched_bunch()

        beam.setup_beam(dt=self.matched_dt, dE=self.matched_dE)

    def build_hamiltonian_model(self):
        """
        Collects symbolic Hamiltonians from each element in ring.
        Adds Hamiltonians together to create 'one turn map'.

        Returns
        -------
        List of symbolic Hamiltonians.
        """

        self.q, self.p = sp.symbols("q p")

        H_list = []
        for element in self.simulation.ring.elements.elements:
            if isinstance(
                element,
                (
                    SingleHarmonicRFStation,
                    DriftSimple,
                ),
            ):
                H_list.append(
                    element.symbolic_hamiltonian(
                        q=self.q,
                        p=self.p,
                        beam=self.beam,
                        ring=self.simulation.ring,
                    )
                )
        self.H_list = H_list
        return self.H_list

    def make_sympy_map(self):
        """Create the symbolic one turn map from the ring to the desired order."""

        self.H_eff = bch_lattice(self.H_list, self.q, self.p, order=self.order)
        self.H_eff = sp.sympify(self.H_eff)
        self.H_func = sp.lambdify((self.q, self.p), self.H_eff, "numpy")

    def find_bucket_level(self):
        """Find the unstable fixed point of the separatrix."""

        rf = None
        for el in self.simulation.ring.elements.elements:
            if isinstance(el, SingleHarmonicRFStation):
                rf = el
                break

        if rf is None:
            raise RuntimeError("No RF station found for bucket computation")

        beta = self.beam.reference.beta
        C = self.simulation.ring.circumference
        omega_rf = 2 * np.pi * rf.harmonic * beta * c0 / C
        phi = rf.phi_rf - np.pi
        q_saddle = (np.pi - phi) / omega_rf
        H_sep = self.H_func(q_saddle, 0.0)

        self.H_sep = float(H_sep)

    def build_numeric_hamiltonian(self):
        """Return numerical evaluation of sympy expression."""
        self.H_func = sp.lambdify((self.q, self.p), self.H_eff, "numpy")

    def sample_matched_bunch(
        self,
        batch_size=10,
    ):
        """
        Sample a distribution that is matched, f(H).

        Parameters
        ----------
        batch_size : int, optional
            Batch size to sample from.
        """
        n = self.n_macroparticles
        matched_dt = np.empty(n, dtype=float)
        matched_dE = np.empty(n, dtype=float)

        count = 0

        with tqdm.tqdm(total=n, desc="Sampling matched bunch") as pbar:
            while count < n:
                q_try = np.random.uniform(
                    self.time_window_limit[0],
                    self.time_window_limit[1],
                    batch_size,
                )
                p_try = np.random.uniform(
                    self.energy_window_limit[0],
                    self.energy_window_limit[1],
                    batch_size,
                )

                H_vals = self.H_func(q_try, p_try)
                mask = H_vals < self.H_sep

                n_accept = mask.sum()
                n_to_take = min(n_accept, n - count)

                if n_to_take > 0:
                    matched_dt[count : count + n_to_take] = q_try[mask][
                        :n_to_take
                    ]
                    matched_dE[count : count + n_to_take] = p_try[mask][
                        :n_to_take
                    ]
                    count += n_to_take
                    pbar.update(n_to_take)

        self.matched_dt = matched_dt
        self.matched_dE = matched_dE

    def apply_desired_emittance(self):
        """Apply desired emittance to algorithm."""
        # not yet sure how to add this to the routine
        pass  # TODO
        # 2d spline interpolation

    def plot_hamiltonian(
        self,
        nq=1000,
        np_=1000,
        q_range=(-1e-9, 1e-9),
        p_range=(-1e6, 1e6),
    ):
        """Plot the Hamiltonian contours."""

        if not hasattr(self, "H_eff"):
            raise RuntimeError("Run make_sympy_map() first")

        H_func = sp.lambdify((self.q, self.p), self.H_eff, "numpy")

        q_vals = np.linspace(*q_range, nq)
        p_vals = np.linspace(
            *p_range,
            np_,
        )

        Q, P = np.meshgrid(q_vals, p_vals)

        H = H_func(Q, P)

        plt.figure(figsize=(6, 5))
        cs = plt.contour(Q, P, H, levels=40)
        plt.clabel(cs, inline=True, fontsize=8)
        plt.xlabel("Δt [s]")
        plt.ylabel("ΔE [eV]")
        plt.title(f"order = {self.order}")
        plt.show()
