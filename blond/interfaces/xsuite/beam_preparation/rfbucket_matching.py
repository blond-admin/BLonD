# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Required scripts for defining the `XsuiteRFBucketMatcher`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import c, e

from blond import SingleHarmonicRFStation
from blond.beam_preparation.base import MatchingRoutine
from blond.core.helpers import int_from_float_with_warning
from blond.physics.drifts import DriftSimple

if TYPE_CHECKING:  # pragma: no cover
    from xpart.longitudinal.rfbucket_matching import (
        ParabolicDistribution,
        QGaussianDistribution,
        ThermalDistribution,
    )

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

    distribution_hints = type[
        ParabolicDistribution | QGaussianDistribution | ThermalDistribution
    ]


class XsuiteRFBucketMatcher(MatchingRoutine):
    """
    Use the XSuite `RFBucketMatcher` for beam matching.

    Beam preparation routine that matches a longitudinal beam distribution
    using the Xsuite RFBucketMatcher and populates the beam with macroparticles.
    REF: https://github.com/xsuite/xsuite.

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
    sigma_z : float, optional
        RMS bunch length, in [m]
        for use in the distribution generation.
    verbose_regeneration : bool, default=False
        Whether to print verbose logs during the matching routine.
    seed : int or None, default=None
        Random seed for reproducible matching.

    Raises
    ------
    ValueError
        If the RF station is not set, energy is not provided, or transition gamma is missing.

    Examples
    --------
    >>> sim.prepare_beam(
    ...     beam= ... ,
    ...     preparation_routine=XsuiteRFBucketMatcher(
    ...         distribution_type=QGaussianDistribution,
    ...         sigma_z= ... ,
    ...         n_macroparticles= ...,
    ...     ),
    ... )
    """

    def __init__(
        self,
        n_macroparticles: int | float,
        distribution_type: distribution_hints,
        sigma_z: float,
        verbose_regeneration: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.distribution_type = distribution_type
        self.sigma_z = sigma_z
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
        )
        self.verbose_regeneration = verbose_regeneration
        self.seed = seed

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Generate and apply a matched longitudinal beam distribution.

        This method constructs an RF bucket from the simulation and rf_station
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
            - The rf_station is not provided.
            - Initial beam energy is not set.
            - No `DriftSimple` elements are found in the ring.
            - `transition_gamma` is not defined in the first drift element.
        """
        # prevent crash if xpart not installed
        from xpart.longitudinal.rf_bucket import RFBucket
        from xpart.longitudinal.rfbucket_matching import RFBucketMatcher

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )

        drift: DriftSimple = simulation.ring.elements.get_element(
            DriftSimple, recursive=False
        )
        rf_station: SingleHarmonicRFStation = (
            simulation.ring.elements.get_element(
                SingleHarmonicRFStation, recursive=False
            )
        )

        if drift.transition_gamma is None:
            raise ValueError(
                "transition_gamma is not set in the first drift element."
            )

        alpha_c = drift.momentum_compaction_factor
        mass_kg = beam.particle_type.mass * e / c**2
        charge_coulomb = beam.particle_type.charge * e

        rfbucket = RFBucket(
            circumference=simulation.ring.circumference,
            gamma=beam.reference.gamma,
            mass_kg=mass_kg,
            charge_coulomb=charge_coulomb,
            alpha_array=np.atleast_1d(alpha_c),
            harmonic_list=np.atleast_1d(rf_station.harmonic),
            voltage_list=np.atleast_1d(rf_station.voltage),
            phi_offset_list=np.atleast_1d(rf_station.phi_rf + np.pi),
            p_increment=0,
        )

        if self.seed is not None:
            np.random.seed(seed=self.seed)  # NOQA
        matcher = RFBucketMatcher(
            rfbucket=rfbucket,
            distribution_type=self.distribution_type,
            sigma_z=self.sigma_z,
            verbose_regeneration=self.verbose_regeneration,
        )

        zeta, delta, *_ = matcher.generate(
            macroparticlenumber=self.n_macroparticles
        )

        omega = rf_station.calc_omega_rf_design(
            beam_beta=beam.reference.beta,
            ring_circumference=simulation.ring.circumference,
        )
        # convert zeta to t coordinate
        T = (2 * np.pi) / omega
        dt = -1 * (zeta) / c + T / 2
        # convert from delta to dE
        dE = delta * beam.reference.total_energy
        beam.setup_beam(
            dt=dt,
            dE=dE,
            mpi_mode="root-distributes",
        )
