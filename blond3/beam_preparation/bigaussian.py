from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import MatchingRoutine
from ..core.backend import backend
from ..core.beam.base import BeamBaseClass
from ..physics.cavities import SingleHarmonicCavity

if TYPE_CHECKING:  # pragma: no cover
    from .. import Simulation


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        sigma_dt: float,
        sigma_dE: StopIteration[float] = None,
        reinsertion: bool=False,
        seed: int=0,
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
        if self._sigma_dE is None:
            sigma_dE = _get_dE_from_dt(ring, rf_station, sigma_dt)  # NOQA TODO
            # IMPORT
        else:
            sigma_dE = self._sigma_dE
        counter = simulation.turn_i.value  # todo might need to be set
        rf_station.rf_program.get_phase(turn_i=counter)
        rf_station.rf_program.get_effective_voltage(turn_i=counter)
        omega_rf = rf_station.omega_rf[0, counter]
        phi_s = rf_station.phi_s[counter]
        phi_rf = rf_station.phi_rf[0, counter]
        eta0 = rf_station.eta_0[counter]

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi


        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = np.random.default_rng(self._seed)
        rng_dE = np.random.default_rng(self._seed + 1)
        beam: BeamBaseClass = simulation._beams[0]

        beam.dt = (
            self._sigma_dt
            * rng_dt.normal(size=beam.n_macroparticles).astype(
                dtype=backend.float, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        beam.dE = sigma_dE * rng_dE.normal(size=beam.n_macroparticles).astype(
            dtype=backend.float, order="C"
        )

        # Re-insert if necessary
        if self._reinsertion:
            while True:
                sel = is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE) == False

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

                beam.dE[sel] = sigma_dE * rng_dE.normal(
                    size=n_new
                ).astype(dtype=backend.float, order="C")
