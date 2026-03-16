# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# References: Simon Lauber
# TODO add original author of bigaussian()

"""Functions needed for :class:`~blond.blond.beam_preparation.bigaussian.BiGaussian`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.analytic.hamilton import (
    calc_phi_s_single_harmonic,
    is_in_separatrix,
)
from blond.beam_preparation.base import MatchingRoutine
from blond.core.helpers import int_from_float_with_warning
from blond.generals.distributed.helpers import (
    mpi_aware_random_generator_cpu,
    mpi_local_size,
)
from blond.generals.iterables_ import all_equal

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from numpy.typing import NDArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

    NumpyArray = NDArray[Any]


def _get_dE_from_dt_core(
    beta: float,
    dt_amplitude: float,
    energy: float,
    eta0: float,
    harmonic: float,
    omega_rf: float,
    particle_charge: float,
    phi_rf: float,
    phi_s: float,
    voltage: float,
) -> float:
    # RF wave is shifted by Pi below transition
    if eta0 < 0:
        phi_rf -= np.pi
    # Calculate dE_amplitude from dt_amplitude using single-harmonic Hamiltonian
    voltage = particle_charge * voltage
    phi_b = omega_rf * dt_amplitude + phi_s
    dE_amplitude = np.sqrt(
        voltage
        * energy
        * beta**2
        * (np.cos(phi_b) - np.cos(phi_s) + (phi_b - phi_s) * np.sin(phi_s))
        / (np.pi * harmonic * np.fabs(eta0))
    )
    return dE_amplitude


def _get_dE_from_dt(
    simulation: Simulation,
    beam: BeamBaseClass,
    dt_amplitude: float,
) -> float:
    r"""
    Evaluate the dE amplitude from dt following a single RF Hamiltonian.

    Parameters
    ----------
    simulation
        Simulation context manager.
    beam
        The Beam object which state will be updated by this element.
    dt_amplitude
        Time amplitude, in [s].

    Returns
    -------
    dE_amplitude
        Full amplitude of the particle oscillation, in [eV].
    """
    from blond.physics.drifts import DriftSimple

    drifts = simulation.ring.elements.get_elements(
        DriftSimple, recursive=False
    )
    above_transition = not simulation.ring.is_below_transition(beam=beam)

    harmonic, omega_rf, phi_rf, voltage = get_main_harmonic_attributes(
        beam=beam,
        simulation=simulation,
    )

    energy = beam.reference.total_energy
    beta = beam.reference.beta

    phi_s = (
        calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=voltage,
            energy_gain=simulation.magnetic_cycle.get_target_total_energy(
                turn_i=0,
                section_i=0,
                reference_time=0,
                particle_type=beam.particle_type,
            )
            - beam.reference.total_energy,
            above_transition=above_transition,
        )
        - phi_rf
    )

    eta0 = [drift.eta_0(gamma=beam.reference.gamma) for drift in drifts]
    assert all_equal(eta0), (
        f"Expected all `eta0` to be the same, but got {eta0}."
    )
    eta0 = eta0[0]

    particle_charge = beam.particle_type.charge

    return _get_dE_from_dt_core(
        beta=float(beta),
        dt_amplitude=dt_amplitude,
        energy=float(energy),
        eta0=eta0,
        harmonic=float(harmonic),
        omega_rf=float(omega_rf),
        particle_charge=particle_charge,
        phi_rf=float(phi_rf),
        phi_s=float(phi_s),
        voltage=float(voltage),
    )


def get_main_harmonic_attributes(
    beam: BeamBaseClass, simulation: Simulation
) -> tuple[float, float, float, float]:
    """
    Relevant main harmonic attributes of all RF stations in :class:`~blond.core.ring.ring.Ring`.

    Parameters
    ----------
    beam
        Simulation :class:`~blond.core.beam.beam.Beam` object.
    simulation
        `Simulation` context manager.

    Returns
    -------
    harmonic
        Main harmonic.
    omega_rf
        Main angular frequency, in [Hz].
    phi_rf
        Main phase, in [rad].
    voltage
        Main voltage, in [V].
    """
    # TODO move this into ring.
    from blond import MultiHarmonicRFStation
    from blond.physics.cavities import SingleHarmonicRFStation

    rf_stations = simulation.ring.elements.get_elements(
        SingleHarmonicRFStation, recursive=False
    ) + simulation.ring.elements.get_elements(
        MultiHarmonicRFStation, recursive=False
    )
    # omega_rf should be all same
    omega_rf = [
        rf.calc_main_harmonic_omega_rf_design(
            beam_beta=beam.reference.beta,
            ring_circumference=simulation.ring.circumference,
        )
        for rf in rf_stations
    ]
    assert all_equal(omega_rf), (
        f"Expected all `omega_rf` to be the same, but got {omega_rf}."
    )
    omega_rf = float(omega_rf[0])

    # phi_rf should be all same
    phi_rf = [rf.get_main_harmonic_phi_rf() for rf in rf_stations]

    assert all_equal(phi_rf), (
        f"Expected all `phi_rf` to be the same, but got {phi_rf}."
    )
    phi_rf = float(phi_rf[0])

    # harmonic should be all same
    harmonic = [rf.get_main_harmonic() for rf in rf_stations]
    assert all_equal(harmonic), (
        f"Expected all `harmonic` to be the same, but got {harmonic}."
    )
    harmonic = float(harmonic[0])

    # voltage sum
    voltage = sum([rf.get_main_harmonic_voltage() for rf in rf_stations])

    return harmonic, omega_rf, phi_rf, voltage


class BiGaussian(MatchingRoutine):
    """
    Beam matching routine to generate a 2D Gaussian particle distribution.

    Parameters
    ----------
    n_macroparticles
        Number of macroparticles to be generated.
    sigma_dt
        Normal distribution length, in [s].
        Effective `sigma_dt` might be smaller, if `reinsertion=True`.
    sigma_dE
        Normal distribution height, in [eV].
        Effective `sigma_dE` might be smaller, if `reinsertion=True`.
    reinsertion
        If True, only particles within the separatrix are generated.
        This affects the effective `sigma_dt` and `sigma_dE`.
    seed
        Random seed parameter.

    Examples
    --------
    >>> from blond import Simulation, BiGaussian
    >>> simulation = Simulation( ... )
    >>> simulation.prepare_beam(
    ...     beam= ... ,
    ...     preparation_routine=BiGaussian( ... ),
    ... )
    """

    def __init__(
        self,
        n_macroparticles: int | float,
        sigma_dt: float,
        sigma_dE: float | None = None,
        reinsertion: bool = False,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self._n_macroparticles_local = mpi_local_size(
            int_from_float_with_warning(
                n_macroparticles, warning_stacklevel=2
            ),
            warning_hint="n_macroparticles",
        )
        self._sigma_dt = sigma_dt
        self._sigma_dE = sigma_dE
        self._reinsertion = reinsertion
        self._seed = seed

    def prepare_beam(
        self,
        simulation: Simulation,
        beam: BeamBaseClass,
    ) -> None:
        """
        Populate the `Beam` object with macro-particles.

        Parameters
        ----------
        simulation
            `Simulation` context manager.
        beam
            Simulation :class:`~blond.core.beam.beam.Beam` object.
        """
        from blond.core.backends.backend import backend
        from blond.physics.drifts import DriftSimple

        super().prepare_beam(
            simulation=simulation,
            beam=beam,
        )
        above_transition = not simulation.ring.is_below_transition(beam=beam)
        harmonic, omega_rf, phi_rf, voltage = get_main_harmonic_attributes(
            beam=beam,
            simulation=simulation,
        )

        drifts: tuple[DriftSimple, ...] = (
            simulation.ring.elements.get_elements(DriftSimple, recursive=False)
        )

        if self._sigma_dE is None:
            sigma_dE = _get_dE_from_dt(
                beam=beam,
                simulation=simulation,
                dt_amplitude=self._sigma_dt,
            )
            # IMPORT
            assert not backend.isnan(sigma_dE), "BUG, fix phi_s"
        else:
            sigma_dE = self._sigma_dE

        phi_s = float(
            calc_phi_s_single_harmonic(
                charge=beam.particle_type.charge,
                voltage=voltage,
                energy_gain=simulation.magnetic_cycle.get_target_total_energy(
                    turn_i=0,
                    section_i=0,
                    reference_time=0,
                    particle_type=beam.particle_type,
                )
                - beam.reference.total_energy,
                above_transition=above_transition,
            )
            - phi_rf
        )
        # call to legacy
        eta0 = [drift.eta_0(gamma=beam.reference.gamma) for drift in drifts]
        assert all_equal(eta0), (
            f"Expected all `eta0` to be the same, but got {eta0}."
        )
        eta0 = eta0[0]

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi

        rng_dt_cpu_only = mpi_aware_random_generator_cpu(
            seed=(self._seed + 0) if self._seed is not None else None,
            n_forward_per_rank=self._n_macroparticles_local,
        )
        rng_dE_cpu_only = mpi_aware_random_generator_cpu(
            seed=(self._seed + 1) if self._seed is not None else None,
            n_forward_per_rank=self._n_macroparticles_local,
        )
        dt = backend.array(  # potentially on GPU
            self._sigma_dt
            * rng_dt_cpu_only.standard_normal(
                size=self._n_macroparticles_local,
                dtype=backend.float,
            )
            + (phi_s - phi_rf) / omega_rf,
            copy=False,
        )
        dE = backend.array(  # potentially on GPU
            sigma_dE
            * rng_dE_cpu_only.standard_normal(
                size=self._n_macroparticles_local,
                dtype=backend.float,
            ),
            copy=False,
        )

        # Re-insert if necessary
        if self._reinsertion:
            while True:
                sel = (
                    is_in_separatrix(
                        charge=beam.particle_type.charge,
                        harmonic=harmonic,
                        voltage=voltage,
                        omega_rf=omega_rf,
                        phi_rf_d=phi_rf,
                        phi_s=phi_s,
                        etas=[eta0],
                        beta=beam.reference.beta,
                        total_energy=beam.reference.total_energy,
                        ring_circumference=simulation.ring.circumference,
                        dt=dt,
                        dE=dE,
                    )
                    == False
                )

                n_new = int(backend.sum(sel))
                if n_new == 0:
                    break
                dt[sel] = backend.array(  # potentially on GPU
                    self._sigma_dt
                    * rng_dt_cpu_only.standard_normal(
                        size=n_new,
                        dtype=backend.float,
                    )
                    + (phi_s - phi_rf) / omega_rf,
                    copy=False,
                )

                dE[sel] = backend.array(  # potentially on GPU
                    sigma_dE
                    * rng_dE_cpu_only.standard_normal(
                        size=n_new,
                        dtype=backend.float,
                    ),
                    copy=False,
                )
        beam.setup_beam(
            dt=dt,
            dE=dE,
            mpi_mode="all-ranks",  # because the random generator above is MPI aware
        )
