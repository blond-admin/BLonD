from __future__ import annotations

from typing import TYPE_CHECKING

from .base import MatchingRoutine
from ..physics.cavities import CavityBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from .. import Simulation


class BiGaussian(MatchingRoutine):
    def __init__(
        self,
        rms_dt: float,
        reinsertion: bool,
        seed: int,
    ):
        super().__init__()
        self._rms_dt = rms_dt
        self._reinsertion = reinsertion
        self._seed = seed

    def on_prepare_beam(
        self,
        simulation: Simulation,
    ) -> None:
        rf_station: CavityBaseClass = simulation.ring.elements.get_element(
            CavityBaseClass
        )
        if self._sigma_dE is None:
            sigma_dE = _get_dE_from_dt(ring, rf_station, sigma_dt)  # NOQA TODO
            # IMPORT
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

        beam.sigma_dt = sigma_dt
        beam.sigma_dE = sigma_dE

        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = np.random.default_rng(seed)
        rng_dE = np.random.default_rng(seed + 1)

        beam.dt = (
            sigma_dt
            * rng_dt.normal(size=beam.n_macroparticles).astype(
                dtype=bm.precision.real_t, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        beam.dE = sigma_dE * rng_dE.normal(size=beam.n_macroparticles).astype(
            dtype=bm.precision.real_t, order="C"
        )

        # Re-insert if necessary
        if reinsertion:
            itemindex = bm.where(
                is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE) == False
            )[0]
            while itemindex.size > 0:
                beam.dt[itemindex] = (
                    sigma_dt
                    * rng_dt.normal(size=itemindex.size).astype(
                        dtype=bm.precision.real_t, order="C", copy=False
                    )
                    + (phi_s - phi_rf) / omega_rf
                )

                beam.dE[itemindex] = sigma_dE * rng_dE.normal(
                    size=itemindex.size
                ).astype(dtype=bm.precision.real_t, order="C")

                itemindex = bm.where(
                    is_in_separatrix(ring, rf_station, beam, beam.dt, beam.dE) == False
                )[0]
