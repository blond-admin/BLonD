# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Fill selected buckets with given emittances and intensities, with a wake.

PS-like ring at 7 GeV/c on h=8, accelerating (sin(phi_s) = 0.5, so the
buckets are tilted), with a broadband resonator. Four of the eight buckets
are filled, each with its own emittance and share of the intensity. The
emittances hold for the matched beam, including the wake.
"""

import numpy as np

from blond import (
    Beam,
    DriftSimple,
    MagneticCyclePerTurn,
    PeriodicFreqSolver,
    Resonators,
    Ring,
    Simulation,
    SingleHarmonicRFStation,
    StaticProfile,
    WakeField,
    momentum_compaction_factor,
    proton,
    setup_backend,
)
from blond.experimental.beam_preparation.bucket_filler_functions import (
    hamilton_to_density_by_emittance,
)
from blond.experimental.beam_preparation.semi_empiric_matcher import (
    SemiEmpiricMatcher,
)
from blond.testing import pytest_active

if not pytest_active():  # pragma: no cover
    setup_backend("auto")

increment_intensity_effects_until_iteration_i = 10


def main():
    # Defining a 1 turn simulation with values typical of the Proton Synchrotron
    phi_rf = 0
    gamma_transition = 6.2
    # 0.1 MeV/c per turn: sin(phi_s) = 0.5 with 200 kV
    momentum_program = np.array([7e9, (7 + 0.0001) * 1e9])
    circumference = 2 * np.pi * 100
    rf_voltage = 200e3
    harmonic = 8
    intensity = 1e13

    # Simulation specific parameters
    number_of_macroparticles = 1e5
    number_of_profile_bins = 1000

    # Defining Some resonator impedance
    Q = 3
    resonator_resonant_frequency = 4e6  # Hz
    resonator_r_shunt = 10000  # Ohms
    profile_length = 2.1e-06

    # Set up simulation
    ring = Ring(circumference=circumference)
    energy_cycle = MagneticCyclePerTurn(
        value_init=float(momentum_program[0]),
        values_after_turn=momentum_program[1:].copy(),
        reference_particle=proton,
    )

    cavity = SingleHarmonicRFStation()
    cavity.harmonic = harmonic
    cavity.voltage = rf_voltage
    cavity.phi_rf_design = phi_rf

    drift = DriftSimple(
        orbit_length=circumference,
        momentum_compaction_factor=momentum_compaction_factor(
            gamma_transition
        ),
    )

    beam = Beam(intensity=intensity, particle_type=proton)

    profile = StaticProfile(0, profile_length, number_of_profile_bins)
    wakefield = WakeField(
        sources=(
            Resonators(
                np.array([resonator_r_shunt]),
                np.array([resonator_resonant_frequency]),
                np.array([Q]),
            ),
        ),
        solver=PeriodicFreqSolver(profile_length, allow_next_fast_len=True),
        profile=profile,
    )

    ring.add_elements(
        (
            cavity,
            drift,
            profile,
            wakefield,
        ),
        reorder=False,
        section_index=0,
    )

    sim = Simulation(ring=ring, magnetic_cycle=energy_cycle)

    t_rf = energy_cycle.get_t_rev_init(circumference, proton) / harmonic
    # bucket k lies around (k + 0.5) * t_rf
    filled_buckets = [0, 2, 5, 6]
    emittances = [1.0, 2.0, 1.0, 3.0]  # 4 pi RMS emittance, [eVs]
    intensity_fractions = [0.3, 0.2, 0.1, 0.4]

    # One window of +- t_rf around each bucket to fill, one dict of
    # arguments of the density function per window, and the relative
    # intensity of each bunch.
    sim.prepare_beam(
        beam=beam,
        preparation_routine=SemiEmpiricMatcher(
            time_limit=[
                ((k - 0.5) * t_rf, (k + 1.5) * t_rf) for k in filled_buckets
            ],
            n_macroparticles=int(number_of_macroparticles),
            hamilton_to_density_function=hamilton_to_density_by_emittance,
            hamilton_to_density_kwargs=[
                dict(emittance_4pi_rms=emittance, exponent=1.0)
                for emittance in emittances
            ],
            bucket_weights=intensity_fractions,
            animate=True,
            tolerance_potential_well=1e-3,
            increment_intensity_effects_until_iteration_i=increment_intensity_effects_until_iteration_i,
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
