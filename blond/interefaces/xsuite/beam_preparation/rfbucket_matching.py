from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union

import numpy as np
from scipy.constants import c, e
from xpart.longitudinal.rf_bucket import RFBucket
from xpart.longitudinal.rfbucket_matching import (
    RFBucketMatcher,
)

from blond import SingleHarmonicCavity
from blond.beam_preparation.base import MatchingRoutine

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Tuple

    from xpart.longitudinal.rfbucket_matching import (
        ParabolicDistribution,
        QGaussianDistribution,
        StationaryDistribution,
        ThermalDistribution,
        WaterbagDistribution,
    )

    from blond._core.beam.base import BeamBaseClass
    from blond._core.simulation.simulation import Simulation

    # TODO tests should cover all generators, if included in type hint
    distribution_hints = Type[
        Union[
            ThermalDistribution,
            ThermalDistribution,
            QGaussianDistribution,
            ParabolicDistribution,
            WaterbagDistribution,
            StationaryDistribution,  # general, in case xsuite was extended
        ]
    ]


class XsuiteRFBucketMatcher(MatchingRoutine):
    def __init__(
        self,
        n_macroparticles: int | float,
        distribution_type: distribution_hints,
        cavity: Optional[SingleHarmonicCavity] = None,
        sigma_z: Optional[float] = None,
        energy_init: Optional[float] = None,
        verbose_regeneration: bool = False,
    ) -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.sigma_z = sigma_z
        self.n_macroparticles = n_macroparticles
        self.verbose_regeneration = verbose_regeneration
        self.energy_init = energy_init
        self.cavity = cavity

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populates the `Beam` object with macro-particles

        Parameters
        ----------
        simulation
            Simulation context manager
        """

        from blond.physics.drifts import DriftSimple

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        drifts: Tuple[DriftSimple, ...] = (
            simulation.ring.elements.get_elements(DriftSimple)
        )
        for _drift in drifts:
            _drift.apply_schedules(
                turn_i=0,
                reference_time=0,
            )
        rf_element = self.cavity
        # TODO what  if rf_element is None
        rf_element.apply_schedules(turn_i=0, reference_time=0)

        energy = self.energy_init
        rest_mass = beam.particle_type.mass
        # TODO what  if energy is None
        gamma = energy / rest_mass
        transition_gamma = drifts[0].transition_gamma
        # TODO what  if transition_gamma is None
        alpha_c = 1 / transition_gamma**2 - 1 / gamma**2
        mass_kg = beam.particle_type.mass * e / c**2
        charge_coulomb = beam.particle_type.charge * e

        # --- Build RF bucket ---
        rfbucket = RFBucket(
            circumference=simulation.ring.circumference,
            gamma=gamma,
            mass_kg=mass_kg,
            charge_coulomb=charge_coulomb,
            alpha_array=np.atleast_1d(alpha_c),
            # TODO what  if cavity is None
            harmonic_list=np.atleast_1d(self.cavity.harmonic),
            voltage_list=np.atleast_1d(self.cavity.voltage),
            phi_offset_list=np.atleast_1d(self.cavity.phi_rf),
            p_increment=0,
        )

        matcher = RFBucketMatcher(
            rfbucket=rfbucket,
            distribution_type=self.distribution_type,
            sigma_z=self.sigma_z,
            verbose_regeneration=self.verbose_regeneration,
        )

        z, delta, *_ = matcher.generate(
            macroparticlenumber=self.n_macroparticles
        )

        print(z, delta)

        # --- Set the beam using standard interface ---
        beam.setup_beam(dt=z / c, dE=delta)
