from __future__ import annotations

from typing import TYPE_CHECKING, Type, Union

import numpy as np
from scipy.constants import c, e
from xpart.longitudinal.rf_bucket import RFBucket
from xpart.longitudinal.rfbucket_matching import (
    RFBucketMatcher,
)

from blond import SingleHarmonicCavity
from blond._core.helpers import int_from_float_with_warning
from blond.beam_preparation.base import MatchingRoutine

if TYPE_CHECKING:  # pragma: no cover
    from typing import Optional, Tuple

    from xpart.longitudinal.rfbucket_matching import (
        ParabolicDistribution,
        QGaussianDistribution,
        ThermalDistribution,
    )

    from blond._core.beam.base import BeamBaseClass
    from blond._core.simulation.simulation import Simulation

    distribution_hints = Type[
        Union[
            ParabolicDistribution,
            QGaussianDistribution,
            ThermalDistribution,
        ]
    ]


class XsuiteRFBucketMatcher(MatchingRoutine):
    """
    Beam preparation routine that matches a longitudinal beam distribution
    using the Xsuite RFBucketMatcher and populates the beam with macroparticles.
    REF:

    This class constructs an RF bucket using the given machine parameters and
    applies a stationary distribution (e.g., Q-Gaussian, Thermal) to initialize
    the beam's longitudinal phase space (`dt`, `dE`) in a matched state.

    Parameters
    ----------
    n_macroparticles : int or float
        Number of macroparticles to generate in the matched distribution.
    distribution_type : type
        Type of stationary distribution to use for matching. Must be a class from
        `xpart.longitudinal.rfbucket_matching`, such as `QGaussianDistribution`
        or `ThermalDistribution`.
    cavity : SingleHarmonicCavity, optional
        RF cavity to use when constructing the RF bucket. Required for voltage,
        harmonic number, and phase.
    sigma_z : float, optional
        RMS bunch length, in [m]
        for use in the distribution generation.
    energy_init : float, optional
        Initial beam energy, in [eV].
        Required for relativistic and bucket parameters.
    verbose_regeneration : bool, default=False
        Whether to print verbose logs during the matching routine.

    Examples
    --------
    >>> sim.prepare_beam(
    >>>     beam= ... ,
    >>>     preparation_routine=XsuiteRFBucketMatcher(
    >>>         distribution_type=QGaussianDistribution,
    >>>         energy_init= ... ,
    >>>         cavity= ...,
    >>>         sigma_z= ... ,
    >>>         n_macroparticles= ...,
    >>>     ),
    >>> )

    Raises
    ------
    ValueError
        If the cavity is not set, energy is not provided, or transition gamma is missing.

    """

    def __init__(
        self,
        n_macroparticles: int | float,
        distribution_type: distribution_hints,
        sigma_z: float,
        verbose_regeneration: bool = False,
    ) -> None:
        super().__init__()
        self.distribution_type = distribution_type
        self.sigma_z = sigma_z
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self.verbose_regeneration = verbose_regeneration

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Generate and apply a matched longitudinal beam distribution.

        This method constructs an RF bucket from the simulation and cavity
        parameters, computes a stationary longitudinal distribution using
        `RFBucketMatcher`, and populates the `Beam` object with macroparticles
        matched to the bucket.

        Parameters
        ----------
        simulation : Simulation
            The simulation context, which includes the ring, drift elements,
            magnetic cycle, and RF systems.
        beam : BeamBaseClass
            The beam to be populated. Must have `particle_type.mass` and
            `particle_type.charge` defined.

        Raises
        ------
        ValueError
            If:
            - The cavity is not provided.
            - Initial beam energy is not set.
            - No `DriftSimple` elements are found in the ring.
            - `transition_gamma` is not defined in the first drift element.

        """
        from blond.physics.drifts import DriftSimple

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        drift: DriftSimple = simulation.ring.elements.get_element(DriftSimple)
        drift.apply_schedules(
            turn_i=0,
            reference_time=0,
        )
        cavity: SingleHarmonicCavity = simulation.ring.elements.get_element(
            SingleHarmonicCavity
        )

        cavity.apply_schedules(turn_i=0, reference_time=0.0)

        if drift.transition_gamma is None:
            raise ValueError(
                "transition_gamma is not set in the first drift element."
            )

        alpha_c = drift.momentum_compaction_factor
        mass_kg = beam.particle_type.mass * e / c**2
        charge_coulomb = beam.particle_type.charge * e

        rfbucket = RFBucket(
            circumference=simulation.ring.circumference,
            gamma=beam.reference_gamma,
            mass_kg=mass_kg,
            charge_coulomb=charge_coulomb,
            alpha_array=np.atleast_1d(alpha_c),
            harmonic_list=np.atleast_1d(cavity.harmonic),
            voltage_list=np.atleast_1d(cavity.voltage),
            phi_offset_list=np.atleast_1d(cavity.phi_rf + np.pi),
            p_increment=0,
        )

        np.random.seed(seed=42)
        matcher = RFBucketMatcher(
            rfbucket=rfbucket,
            distribution_type=self.distribution_type,
            sigma_z=self.sigma_z,
            verbose_regeneration=self.verbose_regeneration,
        )

        zeta, delta, *_ = matcher.generate(
            macroparticlenumber=self.n_macroparticles
        )

        omega = cavity.calc_omega(
            beam_beta=beam.reference_beta,
            ring_circumference=simulation.ring.circumference,
        )
        # convert zeta to t coordinate
        T = (2 * np.pi) / omega
        dt = -1 * (zeta) / c + T / 2
        # convert from delta to dE
        dE = delta * beam.reference_total_energy
        beam.setup_beam(dt=dt, dE=dE)
