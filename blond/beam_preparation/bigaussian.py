# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Functions needed for :class:`~blond.blond.beam_preparation.bigaussian.BiGaussian`.

Authors
-------
Simon Lauber
"""  # TODO add original author of bigaussian()

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.acc_math.analytic.hamilton import (
    calc_phi_s_single_harmonic,
    is_in_separatrix,
)
from blond.beam_preparation.base import MatchingRoutine
from blond.core.backends.backend import backend
from blond.core.helpers import int_from_float_with_warning
from blond.generals.iterables_ import all_equal

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation


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
    r"""Evaluates the dE amplitude from dt following a single RF Hamiltonian.

    Returns
    -------
    dE_amplitude : float
        Full amplitude of the particle oscillation, in [eV]

    """
    from blond.physics.drifts import DriftSimple

    drifts = simulation.ring.elements.get_elements(DriftSimple)
    above_transition = [
        beam.reference_gamma > drift.transition_gamma for drift in drifts
    ]
    assert all_equal(above_transition), (
        f"expected all `above_transition` to be equal, but got {above_transition}"
    )
    above_transition = above_transition[0]

    harmonic, omega_rf, phi_rf, voltage = get_main_harmonic_attributes(
        beam=beam,
        simulation=simulation,
    )

    energy = beam.reference_total_energy
    beta = beam.reference_beta

    phi_s = calc_phi_s_single_harmonic(
        charge=beam.particle_type.charge,
        voltage=voltage,
        phase=phi_rf,
        energy_gain=simulation.magnetic_cycle.get_target_total_energy(
            1, 0, 0, particle_type=beam.particle_type
        )
        - beam.reference_total_energy,
        above_transition=above_transition,
    )

    eta0 = [drift.eta_0(gamma=beam.reference_gamma) for drift in drifts]
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
    """Relevant main harmonic attributes of all RF stations in :class:`~blond.core.ring.ring.Ring`.

    Parameters
    ----------
    simulation
        Simulation context manager
    beam
        Simulation :class:`~blond.core.beam.beam.Beam` object

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
    from blond import MultiHarmonicRfStation
    from blond.physics.cavities import SingleHarmonicRfStation

    rf_stations = simulation.ring.elements.get_elements(
        SingleHarmonicRfStation
    ) + simulation.ring.elements.get_elements(MultiHarmonicRfStation)
    for _rf_station in rf_stations:
        _rf_station.apply_schedules(
            turn_i=0,
            reference_time=0,
        )
    # omega_rf should be all same
    omega_rf = [
        rf.calc_main_harmonic_omega_rf(
            beam_beta=beam.reference_beta,
            ring_circumference=simulation.ring.circumference,
        )
        for rf in rf_stations
    ]
    assert all_equal(omega_rf), (
        f"Expected all `omega_rf` to be the same, but got {omega_rf}."
    )
    omega_rf = float(omega_rf[0])

    # phi_rf should be all same
    try:
        phi_rf = [
            rf.get_main_harmonic_phi_rf()
            + rf.delta_phi_rf[rf.main_harmonic_idx]
            for rf in rf_stations
        ]
    except AttributeError:
        phi_rf = [
            rf.get_main_harmonic_phi_rf() + rf.delta_phi_rf
            for rf in rf_stations
        ]
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
    """Beam matching routine to generate a 2D Gaussian particle distribution.

    Parameters
    ----------
    n_macroparticles
        Number of macroparticles to be generated
    sigma_dt
        Normal distribution length, in [s].
        Effective `sigma_dt` might be smaller, if `reinsertion=True`
    sigma_dE
        Normal distribution height, in [eV].
        Effective `sigma_dE` might be smaller, if `reinsertion=True`
    reinsertion
        If True, only particles within the separatrix are generated.
        This affects the effective `sigma_dt` and `sigma_dE`
    seed
        Random seed parameter

    Examples
    --------
    >>> simulation = Simulation( ... )
    >>> simulation.prepare_beam(
    >>>     beam= ... ,
    >>>     preparation_routine=BiGaussian( ... ),
    >>> )
    """

    def __init__(
        self,
        n_macroparticles: int | float,
        sigma_dt: float,
        sigma_dE: float | None = None,
        reinsertion: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_macroparticles = int_from_float_with_warning(
            n_macroparticles, warning_stacklevel=2
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
        """Populates the `Beam` object with macro-particles.

        Parameters
        ----------
        simulation
            Simulation context manager
        beam
            Simulation :class:`~blond.core.beam.beam.Beam` object

        """
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
            simulation.ring.elements.get_elements(DriftSimple)
        )
        for _drift in drifts:
            _drift.apply_schedules(
                turn_i=0,
                reference_time=0,
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

        phi_s = calc_phi_s_single_harmonic(
            charge=beam.particle_type.charge,
            voltage=voltage,
            phase=phi_rf,
            energy_gain=simulation.magnetic_cycle.get_target_total_energy(
                0, 0, 0, particle_type=beam.particle_type
            )
            - beam.reference_total_energy,
            above_transition=above_transition,
        )
        # call to legacy
        eta0 = [drift.eta_0(gamma=beam.reference_gamma) for drift in drifts]
        assert all_equal(eta0), (
            f"Expected all `eta0` to be the same, but got {eta0}."
        )
        eta0 = eta0[0]

        # RF wave is shifted by Pi below transition
        if eta0 < 0:
            phi_rf -= np.pi

        # Generate coordinates. For reproducibility, a separate random number stream is used for dt and dE
        rng_dt = backend.random.default_rng(self._seed)
        rng_dE = backend.random.default_rng(self._seed + 1)

        dt = (
            self._sigma_dt
            * rng_dt.standard_normal(size=self.n_macroparticles).astype(
                dtype=backend.float, order="C", copy=False
            )
            + (phi_s - phi_rf) / omega_rf
        )
        dE = sigma_dE * rng_dE.standard_normal(
            size=self.n_macroparticles
        ).astype(dtype=backend.float, order="C")

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
                        beta=beam.reference_beta,
                        total_energy=beam.reference_total_energy,
                        ring_circumference=simulation.ring.circumference,
                        dt=dt,
                        dE=dE,
                    )
                    == False
                )

                n_new = int(backend.sum(sel))
                if n_new == 0:
                    break
                dt[sel] = (
                    self._sigma_dt
                    * rng_dt.standard_normal(size=n_new).astype(
                        dtype=backend.float, order="C", copy=False
                    )
                    + (phi_s - phi_rf) / omega_rf
                )

                dE[sel] = sigma_dE * rng_dE.standard_normal(size=n_new).astype(
                    dtype=backend.float, order="C"
                )
        beam.setup_beam(dt=dt, dE=dE)
