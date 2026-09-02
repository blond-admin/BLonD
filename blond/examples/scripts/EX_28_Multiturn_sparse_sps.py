# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Example of how to configure a simulation with sparse multiturn wakefields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from blond import (
    AllowPlotting,
    Beam,
    BiGaussian,
    ConstantMagneticCycle,
    DriftSimple,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    WakeField,
    backend,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.beam_preparation.helpers import make_multibunch_beam
from blond.handle_results.helpers import callers_relative_path
from blond.physics.impedances.base import (
    SupportsVectorFittedModel,
    WakeFieldSource,
)
from blond.physics.impedances.solvers import MultiPoleSparseSolve
from blond.physics.profiles_sparse import EquidistantMultiProfile
from blond.testing import pytest_active

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray

if not pytest_active():  # pragma: no cover
    setup_backend("auto")


class VectorFittedModel(WakeFieldSource, SupportsVectorFittedModel):
    """
    Each pole+residue represents a circuit.

    Parameters
    ----------
    poles
        Complex poles of an equivalent circuit model.
    residues
        Complex residues of an equivalent circuit model.

    See Also
    --------
    MultiPoleSparseSolve: The corresponding wakefield solver.

    References
    ----------
    https://scikit-rf.readthedocs.io/en/latest/tutorials/VectorFitting.html
    """

    def __init__(self, poles, residues):
        assert len(poles) == len(residues), f"{len(poles)=}  {len(residues)=}"
        self.poles = poles
        self.residues = residues
        self._shunt_impedances_counter_rotating = None

    def sort(self, by: str = "residues"):
        """
        Sort the internal data.

        Parameters
        ----------
        by
            'residues' or 'poles'.
        """
        if by == "residues":
            order = np.argsort(np.abs(self.residues))
        elif by == "poles":  # pragma: no cover
            order = np.argsort(np.abs(self.poles))
        else:  # pragma: no cover
            raise NameError(str(by))
        self.poles = self.poles[order]
        self.residues = self.residues[order]

    @staticmethod
    def from_file(loc: str) -> VectorFittedModel:
        """
        Load vector-fitting data from disk.

        Parameters
        ----------
        loc
            Location of file to load from.

        Returns
        -------
        poles
            `Poles`.
        """
        data = np.load(loc)
        residues = data["residues"].flatten()
        poles = data["poles"].flatten()
        print(f"Loaded {len(residues)} wake sources.")
        return VectorFittedModel(poles=poles, residues=residues)

    def plot(self, freq):
        """
        Plot the poles.

        Parameters
        ----------
        freq
            Frequency axis to plot along.

        Returns
        -------
        h
            The reconstructed frequency response plotted.
        """
        omega = 2 * np.pi * freq
        s = 1j * omega
        h = np.zeros_like(s)
        for i in range(len(self.poles)):
            pk = self.poles[i]
            ck = self.residues[i]
            h += ck / (s - pk)
            # A real pole has no implicit complex conjugate (vector-fitting
            # convention): only double-count via the conjugate term for a
            # genuine complex-conjugate-pair pole.
            if np.imag(pk) != 0:
                h += np.conjugate(ck) / (s - np.conjugate(pk))
        plt.subplot(3, 1, 1)
        plt.plot(freq, 20 * np.log10(np.abs(h)))
        plt.subplot(3, 1, 2)
        plt.plot(freq, np.real(h))
        plt.subplot(3, 1, 3)
        plt.plot(freq, np.imag(h))
        return h

    def get_vectorfit(self) -> tuple[NumpyArray, NumpyArray, NumpyArray]:
        """
        Derive the poles and residues as in vector-fitting.

        Returns
        -------
        poles
            The complex poles.
        residues
            The complex residues.
        counterrotation_signs
            Signs of the poles to deal with higher order oscillators
            in counterrotation. Default is ``1``.
        """
        return (
            self.poles,
            self.residues,
            np.ones(len(self.poles), dtype=backend.float),
        )


def main():
    ring = Ring(
        circumference=6911.56,
    )
    magnetic_cycle = ConstantMagneticCycle(
        reference_particle=proton,
        value=25.92e9,
        in_unit="momentum",
    )
    _bunch = Beam(
        intensity=1e10,
        particle_type=proton,
    )
    drift = DriftSimple(
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=22.82177322938192
        ),
        orbit_length=1.0 * ring.circumference,
    )
    rf_station = SingleHarmonicRFStation(
        harmonic=4620,
        voltage=4e6,
        phi_rf=0.0,
    )
    t_rf = (
        magnetic_cycle.get_t_rev_init(
            ring.circumference,
            particle_type=proton,
        )
        / rf_station.harmonic
    )

    filling_pattern = np.zeros(rf_station.harmonic, bool)
    filling_pattern[::10] = 1
    bins_per_profile = 256

    profile = EquidistantMultiProfile(
        filling_pattern=filling_pattern,
        bins_per_profile=bins_per_profile,
    )

    poles = VectorFittedModel.from_file(
        callers_relative_path("resources/1_sps_gen_new.npz", stacklevel=1)
    )
    poles.sort(by="residues")
    poles.plot(freq=np.linspace(0, 1e9, 10_000))

    wakefield = WakeField(
        sources=(poles,),
        solver=MultiPoleSparseSolve(),
        profile=profile,  # type: ignore
    )
    ring.add_elements(
        (
            wakefield,
            # profile,
            rf_station,
            drift,
        )
    )
    sim = Simulation(
        ring=ring,
        magnetic_cycle=magnetic_cycle,
    )

    sim.prepare_beam(
        preparation_routine=BiGaussian(
            sigma_dt=2e-9 / 4,
            seed=1,
            n_macroparticles=1e4,
        ),
        beam=_bunch,
    )

    beam = make_multibunch_beam(
        beam=_bunch,
        n_times=int(rf_station.harmonic // 10),
        t_distance=t_rf * 10,
    )

    cmap = matplotlib.colormaps["plasma"]
    lims = [
        [profile.profiles[-1].cut_left, profile.profiles[-1].cut_right],
        [2 * _bunch._dE.min(), 2 * _bunch._dE.max()],
    ]

    def live_animation(simulation: Simulation, beam: Beam) -> None:
        """
        Plotting utility.

        Parameters
        ----------
        simulation
            `blond.core.simulation.simulation.Simulation` object.
        beam
            `blond.core.beam.beams.Beam` object.
        """
        solver_: MultiPoleSparseSolve = wakefield.solver  # type: ignore

        plt.figure(0)
        plt.cla()
        plt.title("Last Bunch in Train")
        beam.plot_hist2d(bins=(2096, 128), range=lims)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda v, _: f"{v * 1e9:.1f}",
            )
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda v, _: f"{v * 1e-9:.1f}",
            )
        )
        plt.xlabel("Time Offset [ns]")
        plt.ylabel("Energy Offset [GeV]")

        plt.figure(1)
        if simulation.turn_counter.value == 0:
            plt.title("Pole Attenuation per Turn")
            plt.xlabel("Turn")
            plt.ylabel(r"$\mathcal{Re}(r \cdot W)$")
        states = solver_._states[:-1]
        residues = solver_._residues
        with AllowPlotting():  # handle GPU gracefully
            plt.scatter(
                simulation.turn_counter.value * np.ones(len(states[:])),
                np.real(residues[:] * states[:]),
                c=cmap(np.arange(len(states[:]))),
            )

        if not pytest_active():  # pragma: no cover
            plt.draw()
            plt.pause(0.1)

    live_animation.each_turn_i = 10

    sim.run_simulation(
        beams=beam,
        n_turns=3000 if not pytest_active() else 10,
        callbacks=(live_animation,),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
