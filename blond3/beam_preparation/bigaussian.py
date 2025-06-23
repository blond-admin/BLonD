from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from typing import Optional

from .base import MatchingRoutine
from .._core.backends.backend import backend
from .._core.beam.base import BeamBaseClass
from .._core.helpers import safe_index
from ..physics.cavities import SingleHarmonicCavity
from ..physics.drifts import DriftSimple

if TYPE_CHECKING:  # pragma: no cover
    from .._core.simulation.simulation import Simulation


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

    counter = simulation.turn_i.value  # todo might need to be set

    harmonic = safe_index(rf_station.harmonic, counter)
    energy = simulation.energy_cycle.energy[0, counter]
    beta = simulation.energy_cycle.beta[0, counter]
    omega_rf = rf_station.rf_program.omega_rf[counter]
    phi_s = rf_station.phi_s[counter]
    phi_rf = rf_station.rf_program.get_phase(turn_i=counter)
    eta0 = drift.eta_0[counter]

    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi

    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = (simulation.beams[0].particle_type.charge *
               rf_station.rf_program.get_effective_voltage(counter))
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
        sigma_dt: float,
        sigma_dE: Optional[float] = None,
        reinsertion: bool = False,
        seed: int = 0,
    ):
        super().__init__()
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
            sigma_dE = _get_dE_from_dt(simulation=simulation, sigma_dt=self._sigma_dt)
            # IMPORT
        else:
            sigma_dE = self._sigma_dE
        counter = simulation.turn_i.value  # todo might need to be set
        rf_station.rf_program.get_phase(turn_i=counter)
        rf_station.rf_program.get_effective_voltage(turn_i=counter)

        omega_rf = rf_station.rf_program.get_omega(turn_i=counter)
        phi_s = rf_station.phi_s[counter]  # TODO call to legacy
        phi_rf = rf_station.rf_program.get_phase(turn_i=counter)
        eta0 = drift.eta_0[counter]

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi

        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = np.random.default_rng(self._seed)
        rng_dE = np.random.default_rng(self._seed + 1)
        beam: BeamBaseClass = simulation._beams[0]

        beam.dt = (
            self._sigma_dt
            * rng_dt.normal(size=beam._n_particles__init).astype(
                dtype=backend.float, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        beam.dE = sigma_dE * rng_dE.normal(size=beam._n_particles__init).astype(
            dtype=backend.float, order="C"
        )

        # Re-insert if necessary
        if self._reinsertion:
            while True:
                sel = (
                    is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE) == False
                )

                n_new = np.sum(sel)
                if n_new == 0:
                    return
                beam.dt[sel] = (
                    self._sigma_dt
                    * rng_dt.normal(size=n_new).astype(
                        dtype=backend.float, order="C", copy=False
                    )
                    + (phi_s - phi_rf) / omega_rf
                )

                beam.dE[sel] = sigma_dE * rng_dE.normal(size=n_new).astype(
                    dtype=backend.float, order="C"
                )
