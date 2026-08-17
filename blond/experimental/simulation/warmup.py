# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Frozen-beam pre-tracking to equilibrate collective-effect state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tqdm import tqdm  # type: ignore

from blond.generals.distributed import distributed_array
from blond.physics.impedances.base import WakeField
from blond.physics.profiles import ProfileBaseClass

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass
    from blond.core.simulation.simulation import Simulation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BeamShapeSnapshot:
    """Copy of a beam's macroparticle coordinates and intensity."""

    dt: NumpyArray | CupyArray
    dE: NumpyArray | CupyArray
    flags: NumpyArray | CupyArray
    ids: NumpyArray | CupyArray
    intensity: float


def _snapshot_beam_shape(beam: BeamBaseClass) -> _BeamShapeSnapshot:
    return _BeamShapeSnapshot(
        dt=beam._dt.array_local.copy(),
        dE=beam._dE.array_local.copy(),
        flags=beam._flags.array_local.copy(),
        ids=beam._ids.array_local.copy(),
        intensity=beam.intensity,
    )


def _restore_beam_shape(
    beam: BeamBaseClass, snapshot: _BeamShapeSnapshot
) -> None:
    """
    Restore a beam's macroparticle coordinates from a snapshot.

    Writes into the existing `dt`/`dE`/`flags`/`ids` buffers in place when
    their length matches the snapshot, to avoid reallocating (beams can be
    GB-scale, and this is called every warmup turn). Falls back to
    replacing them with fresh arrays only when the length differs, so this
    also correctly undoes array-length changes caused e.g. by
    `purge_flagged_entries` in between the snapshot and the restore.
    """
    if (
        beam._dt.local_size == len(snapshot.dt)
        and beam._dE.local_size == len(snapshot.dE)
        and beam._flags.local_size == len(snapshot.flags)
        and beam._ids.local_size == len(snapshot.ids)
    ):
        beam._dt.array_local[:] = snapshot.dt
        beam._dE.array_local[:] = snapshot.dE
        beam._flags.array_local[:] = snapshot.flags
        beam._ids.array_local[:] = snapshot.ids
    else:
        beam._dt = distributed_array.DistributedArray(snapshot.dt.copy())
        beam._dE = distributed_array.DistributedArray(snapshot.dE.copy())
        beam._flags = distributed_array.DistributedArray(snapshot.flags.copy())
        beam._ids = distributed_array.DistributedArray(snapshot.ids.copy())
    beam.intensity = snapshot.intensity


def warmup(
    simulation: Simulation,
    beam: BeamBaseClass,
    n_turns: int,
    show_progressbar: bool = True,
    verbose: bool = True,
) -> None:
    """
    Pre-track collective-effect state to equilibrium with a frozen bunch.

    Runs ``n_turns`` of the normal per-turn element pipeline while holding
    the macroparticle coordinates fixed (restored after every warmup turn)
    and the ring turn/section position pinned at its starting value (so
    the RF/energy program never advances), so that turn-dependent internal
    state in wakefield solvers (e.g. deques of past profiles, pole-residue
    decay states, multi-turn induced-voltage buffers) and RF/beam
    feedbacks (IIR filters, delay lines) reaches equilibrium with the
    beam's current bunch shape before real multi-turn tracking begins.
    After the first warmup turn, all profiles (whether attached to a
    ``WakeField`` or standalone ring elements) are computed once and held
    static for the remaining turns.

    Useful at injection, where the collective-effect state should already
    be equilibrated at turn 0 instead of drifting there over hundreds of
    real turns.

    Parameters
    ----------
    simulation
        The `Simulation` whose ring elements the beam is warmed up
        through.
    beam
        The beam to warm up. Its macroparticle coordinates are restored to
        their pre-warmup values when this function returns.
    n_turns
        Number of warmup turns to execute.
    show_progressbar
        If True, displays a progress bar. Default is True.
    verbose
        Will print info if True. Default is True.

    See Also
    --------
    blond.core.simulation.simulation.Simulation.prepare_beam : Populate beam with macroparticles.
    blond.core.simulation.simulation.Simulation.run_simulation : Execute the real beam dynamics tracking.

    Notes
    -----
    - This is experimental: unlike the rest of the public API, it may
      change or be removed without warning.
    - This function does not advance ``simulation.turn_counter``,
      ``simulation.section_counter``, or ``beam.reference`` — every
      warmup turn is executed at the turn index (and RF/energy program
      point) that was current when ``warmup()`` was called, and
      ``beam.reference.time``/``beam.reference.total_energy`` are left to
      advance naturally turn-to-turn during warmup (required for correct
      multi-turn wakefield solver bookkeeping), then reset to their
      pre-warmup values once, when this function returns.
    - The beam's macroparticle coordinates (``dt``, ``dE``, ``flags``,
      ``ids``, ``intensity``) are unchanged after this call, as is every
      profile's ``active`` flag.
    - Only wakefield solver / feedback internal state is intentionally
      left mutated by this call — that is the point of the function.
    - Not integrated with beam-matching routines (e.g.
      ``SemiEmpiricMatcher``) — call this after ``prepare_beam()`` and
      before ``run_simulation()``.
    """
    if n_turns <= 0:
        return

    if verbose:
        logger.info(f"Warming up for {n_turns} turns...")

    simulation.finalize(beams=(beam,), n_turns=n_turns)

    shape_snapshot = _snapshot_beam_shape(beam)
    start_time = beam.reference.time
    start_total_energy = beam.reference.total_energy
    start_turn = simulation.turn_counter.value
    start_section = simulation.section_counter.value

    # Gate on `profile.active`, not `wakefield.track_profile`: a
    # WakeField's profile may also be independently registered as its own
    # ring element (tracked on its own, outside the WakeField's `_track`),
    # and only `profile.active` is checked by both paths.
    # `wakefield.update_induced_voltage` is deliberately left untouched —
    # the solver state must keep updating every warmup turn. Profiles not
    # attached to any WakeField are frozen too, for consistency with "the
    # bunch shape is fixed for the whole warmup".
    wakefields = simulation.ring.elements.get_elements(
        WakeField, recursive=True
    )
    standalone_profiles = simulation.ring.elements.get_elements(
        ProfileBaseClass, recursive=True
    )
    profiles = {wakefield.profile for wakefield in wakefields} | set(
        standalone_profiles
    )
    original_profile_active = {profile: profile.active for profile in profiles}

    iterator = range(n_turns)
    if show_progressbar:
        iterator = tqdm(iterator, desc="BLonD3 warmup")

    try:
        for i in iterator:
            simulation.turn_counter.value = start_turn
            simulation.section_counter.value = start_section

            simulation.mainloop(
                beams=(beam,),
                n_turns=1,
                show_progressbar=False,
            )

            _restore_beam_shape(beam, shape_snapshot)

            if i == 0:
                for profile in profiles:
                    profile.active = False
    finally:
        simulation.turn_counter.value = start_turn
        simulation.section_counter.value = start_section
        beam.reference.time = start_time
        beam.reference.total_energy = start_total_energy
        for profile, was_active in original_profile_active.items():
            profile.active = was_active
