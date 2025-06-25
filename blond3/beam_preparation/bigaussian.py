from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING

import numpy as np

from .base import MatchingRoutine, BeamPreparationRoutine
from .._core.backends.backend import backend
from ..physics.drifts import DriftSimple
from ..physics.cavities import SingleHarmonicCavity

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional

    from .._core.simulation.simulation import Simulation
    from .._core.beam.base import BeamBaseClass
    from .. import RfStationParams


def _get_dE_from_dt(simulation: Simulation, dt_amplitude: float) -> float:
    r"""A routine to evaluate the dE amplitude from dt following a single
    RF Hamiltonian.

    Parameters
    ----------
    ring : class
        A Ring type class
    rf_station : class
        An RFStation type class
    dt_amplitude : float
        Full amplitude of the particle oscillation in [s]

    Returns
    -------
    dE_amplitude : float
        Full amplitude of the particle oscillation in [eV]

    """
    drift: DriftSimple = simulation.ring.elements.get_element(DriftSimple)
    rf_station: SingleHarmonicCavity = simulation.ring.elements.get_element(
        SingleHarmonicCavity
    )
    rf_station_program: RfStationParams = rf_station.rf_program

    counter = simulation.turn_i.value  # todo might need to be set

    harmonic = rf_station_program.harmonic[0, counter]
    energy = simulation.energy_cycle.energy[0, counter]
    beta = simulation.energy_cycle.beta[0, counter]
    omega_rf = rf_station_program.omega_rf[0, counter]
    phi_rf = rf_station_program.phi_rf[0, counter]
    phi_s = np.deg2rad(30 + 180)  # TODO rf_station.phi_s[counter]
    eta0 = drift.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = (
        simulation.beams[0].particle_type.charge
        * rf_station_program.voltage[0, counter]
    )
    eta0 = drift.eta_0[counter]

    phi_b = omega_rf * dt_amplitude + phi_s
    dE_amplitude = np.sqrt(
        voltage
        * energy
        * beta**2
        * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s))
        / (np.pi * harmonic * np.fabs(eta0))
    )

    return dE_amplitude


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        n_macroparticles: int,
        sigma_dt: float,
        sigma_dE: Optional[float] = None,
        reinsertion: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.n_macroparticles = n_macroparticles
        self._sigma_dt = sigma_dt
        self._sigma_dE = sigma_dE
        self._reinsertion = reinsertion
        self._seed = seed

    def on_prepare_beam(
        self,
        simulation: Simulation,
    ) -> None:
        rf_station: SingleHarmonicCavity = simulation.ring.elements.get_element(
            SingleHarmonicCavity
        )
        drift: DriftSimple = simulation.ring.elements.get_element(DriftSimple)

        if self._sigma_dE is None:
            sigma_dE = _get_dE_from_dt(
                simulation=simulation, dt_amplitude=self._sigma_dt
            )
            # IMPORT
        else:
            sigma_dE = self._sigma_dE
        counter = simulation.turn_i.value  # todo might need to be set

        omega_rf = rf_station.rf_program.omega_rf[0, counter]
        phi_rf = rf_station.rf_program.phi_rf[0, counter]
        phi_s = np.deg2rad(30 + 180)  # TODO rf_station.phi_s[counter]  # TODO
        # call to
        # legacy
        eta0 = drift.eta_0[counter]

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi

        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = np.random.default_rng(self._seed)
        rng_dE = np.random.default_rng(self._seed + 1)
        beam: BeamBaseClass = simulation._beams[0]

        dt = (
            self._sigma_dt
            * rng_dt.normal(size=self.n_macroparticles).astype(
                dtype=backend.float, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        dE = sigma_dE * rng_dE.normal(size=self.n_macroparticles).astype(
            dtype=backend.float, order="C"
        )

        # Re-insert if necessary
        if self._reinsertion:
            while True:
                sel = is_in_separatrix(ring, rf_station, beam, dt, dE) == False

                n_new = np.sum(sel)
                if n_new == 0:
                    return
                dt[sel] = (
                    self._sigma_dt
                    * rng_dt.normal(size=n_new).astype(
                        dtype=backend.float, order="C", copy=False
                    )
                    + (phi_s - phi_rf) / omega_rf
                )

                dE[sel] = sigma_dE * rng_dE.normal(size=n_new).astype(
                    dtype=backend.float, order="C"
                )
        beam.setup_beam(dt=dt, dE=dE)
