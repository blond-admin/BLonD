# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Holds `PythonSpecials` and helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.backends.backend import STATE_LAG_BINS as _STATE_LAG_BINS
from blond.core.backends.backend import Specials
from blond.core.beam.flags import BeamFlags

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NumpyArray


# The function definition is recycled by the numba backend.
def _move_flagged_elements_to_end_py(
    flag: int,
    flags: NumpyArray,  # also purged
    dt: NumpyArray,
    dE: NumpyArray,
    ids: NumpyArray,
):
    """
    Reorder entries where ``flags == flag`` to the array end.

    This is only intended for `purge_flagged_entries`.

    Parameters
    ----------
    flag
        The flag to be used as a selector what to place at the end.
    flags
        Macro-particle flags.
    dt
        Macro-particle time coordinates [s].
    dE
        Macro-particle energy coordinates [eV].
    ids
        Macro-particle ids.
        This allows to identify single particles,
        even if the array indexing is changed.

    Returns
    -------
    n_new
        Number of particles that are not flagged.
    """
    i = 0
    j = flags.size - 1

    while i <= j:
        if flags[i] != flag:
            i += 1
        else:
            # flags[i] is True, swap with flags[j]
            flags[i], flags[j] = flags[j], flags[i]
            dt[i], dt[j] = dt[j], dt[i]
            dE[i], dE[j] = dE[j], dE[i]
            ids[i], ids[j] = ids[j], ids[i]
            j -= 1
    return j + 1


def _twc_fir_advance_gap(
    state: complex,
    taper: complex,
    n_free: int,
    phase: float,
    rotation: complex,
) -> tuple[complex, complex]:
    """
    Advance the TWC FIR recursion over event-free lattice sites.

    Closed form of `n_free` repetitions of the single-site update
    ``taper *= rotation`` followed by ``state = (state - taper) *
    rotation`` (no injections, removals or outputs at these sites).

    Parameters
    ----------
    state
        Phasor carrying the untapered cosine wake.
    taper
        Sliding-window accumulator building the linear taper.
    n_free
        Number of event-free lattice sites to advance.
    phase
        Phase advance per lattice site, in [rad].
    rotation
        ``exp(1j * phase)``.

    Returns
    -------
    state, taper
        The advanced phasors.
    """
    rot_gap = np.exp(1j * phase * n_free)
    taper = taper * rot_gap
    state = state * rot_gap - n_free * taper * rotation
    return state, taper


def _twc_fir_one_mode(
    profile: NumpyArray,
    grid_index: NumpyArray,
    r_shunt: float,
    a_tilde: float,
    omega_r: float,
    bin_dt: float,
    two_factor: float,
    voltage: NumpyArray,
) -> None:
    """
    Accumulate one TWC mode of ``wake_from_twc_fir`` into `voltage`.

    See the ``Specials`` ABC for the algorithm and the lattice-grid
    convention of `grid_index`.

    Parameters
    ----------
    profile
        Beam profile histogram (occupied lattice sites only).
    grid_index
        Lattice site of each profile bin, strictly increasing.
    r_shunt
        Shunt impedance of the mode, in [Ohm].
    a_tilde
        Wake support (filling) time of the mode, in [s].
    omega_r
        Angular resonant frequency of the mode, in [rad/s].
    bin_dt
        Spacing of the underlying equidistant lattice, in [s].
    two_factor
        Twice the profile-to-current conversion factor.
    voltage
        Output voltage, in [V]. Accumulated into.
    """
    # W(0) amplitude of the single-cosine kernel (no conjugate pair);
    # the trapezoidal half/half injection below supplies the
    # (sign(t) + 1) factor of the analytic wake
    wake_amplitude = 2 * r_shunt / a_tilde
    # taper length in lattice steps; ceil quantizes the wake support to
    # the lattice (relative error ~ bin_dt / a_tilde)
    n_taper = int(np.ceil(a_tilde / bin_dt))
    inject_scale = two_factor / n_taper
    phase = omega_r * bin_dt
    rotation = np.exp(1j * phase)
    # phase accumulated by a taper term over its full lifetime
    rotation_removal = np.exp(1j * phase * n_taper)

    state = 0.0 + 0.0j
    taper = 0.0 + 0.0j
    # oldest not-yet-removed taper term; term `j` (injected at site
    # `grid_index[j] + 1`) expires at site `grid_index[j] + n_taper + 1`,
    # monotonically in `j`
    oldest = 0
    for bin_i in range(len(profile)):
        if bin_i > 0:
            site = int(grid_index[bin_i - 1])
            target = int(grid_index[bin_i])
            inject_site = site + 1
            inject_pending = True

            while True:
                # earliest event site: the injection comes first (it is
                # at `site + 1`); afterwards the next removal, capped by
                # `target`
                event_site = target
                if inject_pending:
                    event_site = inject_site
                elif oldest < bin_i:
                    expiry = int(grid_index[oldest]) + n_taper + 1
                    event_site = min(expiry, event_site)
                if event_site >= target:
                    break
                if event_site - site > 1:
                    state, taper = _twc_fir_advance_gap(
                        state, taper, event_site - 1 - site, phase, rotation
                    )
                if inject_pending and event_site == inject_site:
                    taper += profile[bin_i - 1] * inject_scale
                    inject_pending = False
                while (
                    oldest < bin_i
                    and int(grid_index[oldest]) + n_taper + 1 == event_site
                ):
                    # fully decayed: remove with the phase accumulated
                    # over its n_taper rotations
                    taper -= profile[oldest] * inject_scale * rotation_removal
                    oldest += 1
                taper *= rotation
                state -= taper
                state *= rotation
                site = event_site

            if target - site > 1:
                state, taper = _twc_fir_advance_gap(
                    state, taper, target - 1 - site, phase, rotation
                )
            # the output site itself: events, then the taper rotation and
            # state subtraction; its state rotation happens after the
            # output below
            if inject_pending and inject_site == target:
                taper += profile[bin_i - 1] * inject_scale
            while (
                oldest < bin_i
                and int(grid_index[oldest]) + n_taper + 1 == target
            ):
                taper -= profile[oldest] * inject_scale * rotation_removal
                oldest += 1
            taper *= rotation
            state -= taper

        profile_i_half = 0.5 * profile[bin_i] * two_factor
        state += profile_i_half
        voltage[bin_i] += wake_amplitude * np.real(state)
        state += profile_i_half

        state *= rotation


class PythonSpecials(Specials):
    """Implementation of backend functions in Python."""

    @staticmethod
    def get_max_threads() -> int:  # pragma: no cover
        """
        Return the max number of threads this backend's kernels may use.

        Returns
        -------
        max_threads
            Maximum number of threads this backend's kernels may use.
        """
        return 1

    @staticmethod
    def beam_phase(
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        """
        Calculate the beam phase.

        Parameters
        ----------
        hist_x
            X axis of the histogram, usually in [s].
        hist_y
            Y axis of the histogram.
        alpha
            # TODO ported from blond2, was undocumented.
        omega_rf
            # TODO ported from blond2, was undocumented.
        phi_rf
            # TODO ported from blond2, was undocumented.
        bin_size
            # TODO ported from blond2, was undocumented.

        Returns
        -------
        beam_phase
            # TODO ported from blond2, was undocumented.
        """
        scoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.sin(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )
        ccoeff = np.trapezoid(  # type: ignore
            np.exp(alpha * hist_x)
            * np.cos(omega_rf * hist_x + phi_rf)
            * hist_y,
            dx=bin_size,
        )

        return scoeff / ccoeff

    @staticmethod
    def histogram(
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: float,
        stop: float,
    ) -> None:
        """
        Calculate the histogram of an array.

        Parameters
        ----------
        array_read
            Array of many entries that should be compressed to a histogram.
        array_write
            Memory of where to write the histogram.
        start
            Start of the histogram bins.
        stop
            Stop of the histogram bins.
        """
        array_write[:], _ = np.histogram(
            array_read,
            range=(float(start), float(stop)),
            bins=len(array_write),
        )

    @staticmethod
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: NumpyArray,
        dE: NumpyArray,
        flags: NumpyArray,
    ) -> None:
        # select particles outside box
        select = (dE > e_max) | (dE < e_min) | (dt < t_min) | (dt > t_max)
        flags[select] = BeamFlags.LOST.value

    @staticmethod
    def kick_single_harmonic(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        """
        Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            RF voltage of the RF station, in [V].
        omega_rf
            Angular frequency of the RF system, in [rad/s].
        phi_rf
            RF station's design phase (per harmonic) in [rad].
        charge
            Particle charge, as number of elementary charges `e` [].
        acceleration_kick
            Energy that is added to all particles, in [eV].
        """
        voltage_kick = charge * voltage

        dE[:] += (
            voltage_kick * np.sin(omega_rf * dt[:] + phi_rf)
            + acceleration_kick
        )

    @staticmethod
    def kick_multi_harmonic(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        """
        Apply ``dE += .. * sin(.. * dt + ..)``.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            RF voltages of the RF station, in [V].
        omega_rf
            Angular frequencies of the RF system, in [rad/s].
        phi_rf
            RF station's design phases (per harmonic) in [rad].
        charge
            Particle charge, as number of elementary charges `e` [].
        n_rf
            Number of RF systems.
        acceleration_kick
            Energy that is added to all particles, in [eV].
        """
        voltage_kick = charge * voltage

        for j in range(n_rf):
            dE += voltage_kick[j] * np.sin(omega_rf[j] * dt + phi_rf[j])

        dE[:] += acceleration_kick

    @staticmethod
    def sum_1d_array(array: NumpyArray) -> float:
        """
        Return the sum of an 1d array.

        Parameters
        ----------
        array
            Input array 1.

        Returns
        -------
        sum_1d_array
            Sum of a 1d arrays.
        """
        return np.sum(array)

    @staticmethod
    def dot_product_1d_array(array_1: NumpyArray, array_2: NumpyArray):
        """
        Return the sum of dot product of two 1d arrays.

        Parameters
        ----------
        array_1
            Input array 1.
        array_2
            Input array 2.

        Returns
        -------
        dot_product_1d_array
            Dot product of two 1d arrays.
        """
        return np.dot(array_1, array_2)

    @staticmethod
    def drift_simple(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        r"""
        Function to apply drift equation of motion.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        T
            Time spend in the drift region, in [s].
            :math:`T = L / (\beta c_0)`.
        eta_0
            General synchrotron parameter (zeroth-order slippage factor) [unitless].
        beta
            Relativistic velocity factor :math:`\beta = v/c` [unitless].
        energy
            Total beam energy [eV].
        """
        # solver_decoded = solver.decode(encoding='utf_8')

        coeff = eta_0 / (beta * beta * energy)
        dt += T * coeff * dE

    @staticmethod
    def drift_exact(
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        higher_alpha: NumpyArray,
        beta: float,
        energy: float,
    ) -> None:  # pragma: no cover
        r"""
        Exact drift equation of motion with higher order momentum compaction factors.

        Parameters
        ----------
        dt : NumpyArray
            Macro-particle time coordinates, in [s].
        dE : NumpyArray
            Macro-particle energy coordinates, in [eV].
        T : float
            Revolution period, in [s].
        alpha_0 : float
            Momentum compaction factor [unitless].
        higher_alpha : NumpyArray
            Momentum compaction factor to higher orders.
        beta
            Relativistic velocity factor :math:\beta = v/c [unitless].
        energy
            Total beam energy [eV].
        """
        n_alpha = len(higher_alpha)
        invbetasq = 1.0 / (beta * beta)
        inv_energy = 1.0 / energy
        inv_energy_sq = inv_energy * inv_energy

        # delta (vectorized)
        beam_delta = (
            np.sqrt(
                1.0
                + invbetasq * (dE * dE * inv_energy_sq + 2.0 * dE * inv_energy)
            )
            - 1.0
        )

        # ---- Polynomial evaluation ----
        poly = 1.0 + alpha_0 * beam_delta

        if n_alpha > 0:
            delta_power = beam_delta * beam_delta  # δ²

            for k in range(n_alpha):
                poly += higher_alpha[k] * delta_power
                delta_power *= beam_delta  # next power

        # ---- Final update ----
        dt += T * (poly * (1.0 + dE * inv_energy) / (1.0 + beam_delta) - 1.0)

    @staticmethod
    def kick_interpolated(
        dt: NumpyArray,
        dE: NumpyArray,
        voltage: NumpyArray,
        bin_centers: NumpyArray,
        charge: float,
        acceleration_kick: float,
        first_left_cut: float | None = None,
        left_cut_distance: float | None = None,
        cut_width: float | None = None,
        bins_per_profile: int | None = None,
        filling_pattern: NumpyArray | None = None,
        bucket_index_to_memory_index: NumpyArray | None = None,
    ) -> None:
        """
        Interpolated kick method.

        Parameters
        ----------
        dt
            Macro-particle time coordinates, in [s].
        dE
            Macro-particle energy coordinates, in [eV].
        voltage
            Array of voltages along `bin_centers`, in [V].
        bin_centers
            Positions of `voltage`, in [s].
        charge
            Particle charge, as number of elementary charges `e` [].
        acceleration_kick
            Energy, in [eV], which is added to all particles.
            This is intended to subtract the target energy from the RF
            energy gain in one common call.
        first_left_cut
            Left edge of the first bucket's histogram. Pass this together
            with the other sparse-metadata arguments below (e.g. via
            `EquidistantMultiProfile.sparse_kick_metadata`) when
            `bin_centers` is a gapped, multi-island array such as
            `EquidistantMultiProfile.hist_x`. When omitted, `bin_centers`
            must be uniformly spaced.
        left_cut_distance
            Distance between the left edge of each bucket's histogram.
        cut_width
            Distance between left and right edge of one bucket's
            histogram.
        bins_per_profile
            Number of bins per bucket.
        filling_pattern
            Filling pattern as a boolean array where `True` means filled
            bucket.
        bucket_index_to_memory_index
            Maps bucket index to memory index, see
            `_gen_array_bucket_index_to_memory_index`.
        """
        sparse = first_left_cut is not None
        n_slices = len(bin_centers)

        if sparse:
            inv_bin_width = bins_per_profile / cut_width
        else:
            if n_slices >= 2:  # noqa: PLR2004
                diffs = np.diff(bin_centers)
                if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=0.0):
                    raise ValueError(
                        "bin_centers is not uniformly spaced (looks like "
                        "a sparse/multi-island "
                        "EquidistantMultiProfile.hist_x). Either pass "
                        "this profile's sparse metadata (first_left_cut, "
                        "left_cut_distance, cut_width, bins_per_profile, "
                        "filling_pattern, bucket_index_to_memory_index), "
                        "e.g. via `profile.sparse_kick_metadata`, or use "
                        "EquidistantMultiProfile.profiles[i].hist_x for "
                        "a single bucket."
                    )
            inv_bin_width = (n_slices - 1) / (bin_centers[-1] - bin_centers[0])

        helper1 = charge * (voltage[1:] - voltage[:-1]) * inv_bin_width
        helper2 = (
            charge * voltage[:-1] - bin_centers[:-1] * helper1
        ) + acceleration_kick

        if not sparse:
            fbin = np.floor((dt - bin_centers[0]) * inv_bin_width).astype(
                np.int32
            )
            for i in range(len(dt)):
                if (fbin[i] >= 0) and (fbin[i] < n_slices - 1):
                    dE[i] += dt[i] * helper1[fbin[i]] + helper2[fbin[i]]
            return

        n_buckets = len(filling_pattern)
        inv_hist_dist = 1.0 / left_cut_distance
        bin_width = cut_width / bins_per_profile
        for i in range(len(dt)):
            bucket_i = int(np.floor((dt[i] - first_left_cut) * inv_hist_dist))
            if bucket_i < 0 or bucket_i >= n_buckets:
                continue
            if not filling_pattern[bucket_i]:
                continue
            cut_left = first_left_cut + bucket_i * left_cut_distance
            bucket_bin_center0 = cut_left + bin_width / 2.0
            local_bin = int(
                np.floor((dt[i] - bucket_bin_center0) * inv_bin_width)
            )
            if local_bin < 0 or local_bin >= bins_per_profile - 1:
                continue
            fbin = bucket_index_to_memory_index[bucket_i] + local_bin
            dE[i] += dt[i] * helper1[fbin] + helper2[fbin]

    @staticmethod
    def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
        beam_dE: NumpyArray,
        energy_lost: float,
        longitudinal_damping_time: float,
        natural_energy_spread: float,
        total_energy: float,
        disable_quantum_excitation: bool = False,
    ) -> None:
        """
        Apply synchrotron radiation and quantum excitation energy kicks.

        Parameters
        ----------
        beam_dE
            Macro-particle energy coordinates, in [eV]. Modified in place.
        energy_lost
            Energy lost through the considered synchrotron segment,
            in [eV per turn].
        longitudinal_damping_time
            Longitudinal damping time of the considered synchrotron segment,
            in [turn].
        natural_energy_spread
            Natural energy spread of the considered synchrotron segment,
            [dimensionless].
        total_energy
            Beam total reference energy, in [eV].
        disable_quantum_excitation
           Disables the quantum excitation kick.
        """
        damping_factor = 1.0 - 2.0 / longitudinal_damping_time
        if disable_quantum_excitation:
            beam_dE *= damping_factor
            beam_dE -= energy_lost
        else:
            noise_scale = (
                2.0
                * natural_energy_spread
                / float(np.sqrt(longitudinal_damping_time))
                * total_energy
            )
            # Pre-combine the additive term in the noise buffer so that the
            # final update over beam_dE is a single fused-multiply-add-like
            # expression: beam_dE := damping_factor * beam_dE + noise_term.
            # Legacy `np.random.standard_normal` is intentional: keeps
            # `np.random.seed(...)` reproducibility on the Python reference
            # backend.
            noise_term = np.random.standard_normal(size=len(beam_dE))  # NOQA: NPY002
            noise_term *= noise_scale
            noise_term -= energy_lost
            # One sweep on beam_dE: scale then add the prepared noise_term.
            beam_dE *= damping_factor
            beam_dE += noise_term

    @staticmethod
    def move_flagged_elements_to_end(
        flag: int,
        flags: NumpyArray,  # also purged
        dt: NumpyArray,
        dE: NumpyArray,
        ids: NumpyArray,
    ):
        """
        Reorder entries where ``flags == flag`` to the array end.

        This is only intended for `purge_flagged_entries`.

        Parameters
        ----------
        flag
            The flag to be used as a selector what to place at the end.
        flags
            Macro-particle flags.
        dt
            Macro-particle time coordinates [s].
        dE
            Macro-particle energy coordinates [eV].
        ids
            Macro-particle ids.
            This allows to identify single particles,
            even if the array indexing is changed.

        Returns
        -------
        n_new
            Number of particles that are not flagged.
        """
        n_new = _move_flagged_elements_to_end_py(
            flag=np.int32(flag),
            flags=flags,
            dt=dt,
            dE=dE,
            ids=ids,
        )
        return n_new

    @staticmethod
    def histogram_sparse(
        x: NumpyArray,
        out: NumpyArray,
        first_left_cut: float,
        left_cut_distance: float,
        cut_width: float,
        bins_per_profile: int,
        n_active_profiles: int,
        filling_pattern: NumpyArray,
        bucket_index_to_memory_index: NumpyArray,
    ) -> None:
        """
        Sparse histogram with strided memory layout (gaps between profiles).

        Parameters
        ----------
        x
            An array, e.g., the particle ``dt`` values.
        out
            Output histogram ``(n_filled_buckets * bins_per_profile)``.
        first_left_cut
            Start of the first histogram.
        left_cut_distance
            Distance between the start of each histogram.
        cut_width
            Distance between left and right edge of the histogram.
        bins_per_profile
            Number of bins per bucket.
        n_active_profiles
            Number of non-empty buckets.
        filling_pattern
            Filling pattern as a boolean array
            where ``True`` means filled bucket.
        bucket_index_to_memory_index
            Maps bucket index to memory index.
            For a ``filling_pattern = [1, 0, 0, 1]``
            ``bucket_index_to_memory_index = [0, 0, 0, 8]`` with
            ``bins_per_profile = 8``.
            Use `_gen_array_bucket_index_to_memory_index` to generate this.
        """
        out[:] = 0
        for bucket_i, active in enumerate(filling_pattern):
            if not active:
                continue
            memory_i = bucket_index_to_memory_index[bucket_i]
            sel = slice(
                memory_i,
                memory_i + bins_per_profile,
            )
            hist, _ = np.histogram(
                x,
                bins=bins_per_profile,
                range=(
                    first_left_cut + bucket_i * left_cut_distance,
                    first_left_cut + bucket_i * left_cut_distance + cut_width,
                ),
            )
            out[sel] = hist

    @staticmethod
    def wake_from_pole_residue(  # NOQA PLR0915
        # read
        profile: NumpyArray,
        profile_dts: NumpyArray,
        poles: NumpyArray,
        residues: NumpyArray,
        is_counterrotating_beam: bool,
        counterrotating_pole_signs: NumpyArray,
        update_on_bin: NumpyArray,
        factor: float,
        # write
        states: NumpyArray,
        voltage: NumpyArray,
        voltage_threaded: NumpyArray,
    ) -> None:
        """
        Apply poles based on the `profile` to generate `voltage`.

        See `Specials.wake_from_pole_residue` for the full derivation of the
        lag bookkeeping and the ``states`` layout; this is the readable
        reference implementation the other backends must match.

        Parameters
        ----------
        profile
            Beam profile histogram.
        profile_dts
            Base for time step, connected to `update_on_bin`.
        poles
            Complex poles of an equivalent circuit model.
        residues
            Complex residues of an equivalent circuit model.
        is_counterrotating_beam
            If true, the current beam is counter-rotating.
        counterrotating_pole_signs
            Array per pole, -1 if the sign of the impedance is flipped
            for a counter-rotating beam.
        update_on_bin
            Index when to trigger an update of dt. For speedup.
            E.g. For profile no.: `0,0,0,1,1,1,1,2,2,2`
            one needs `update_on_bin = [0,3,7]`.
        factor
            To convert `profile` to current per bin [A].
        states
            Complex state vector of length ``2 * n_poles + 2``, initially
            ``(0 + 0j)``. ``states[:n_poles]`` holds each pole's state
            through the last bin, referenced at the time in ``states[-1]``;
            ``states[n_poles:2 * n_poles]`` holds the same state one bin
            earlier, referenced at ``states[-2]``. Both reference times live
            in the real part and are written by this function.
        voltage
            Output voltage, in [V].
        voltage_threaded
            Cached `voltage` array per thread. For speedup.
        """
        n_poles = len(poles)
        two_factor = 2 * factor
        n_bins = len(profile)

        assert len(states) == _STATE_LAG_BINS * (n_poles + 1)
        assert n_bins >= _STATE_LAG_BINS

        voltage[:] = 0
        voltage_threaded[:, :] = 0

        # Reference times of the two incoming states: `t_state` belongs to
        # `states[pole_i]`, `t_state_prev` to the one-bin-older
        # `states[n_poles + pole_i]`.
        t_state = states[-1].real
        t_state_prev = states[-2].real
        # The state a bin reads is referenced two bins back, while the
        # bin-averaged wake starts three half-bins back (the residues carry
        # ((exp(p*dt) - 1) / (p*dt))**3 * exp(p*dt/2)). Every bin is `bin_dt`
        # wide -- a sparse profile's gaps are whole numbers of bins -- so
        # that lookback is the same everywhere.
        bin_dt = profile_dts[1] - profile_dts[0]

        for pole_i in range(n_poles):
            # `cr_pole_flip` is intentionally applied to BOTH the state
            # injection and the output amplitude: for the counter-rotating
            # beam's own wake the two factors cancel (flip**2 == 1); only
            # contributions of the other beam, accumulated in the shared
            # `states`, see a net sign flip.
            cr_pole_flip = 1.0
            if (
                is_counterrotating_beam
                and counterrotating_pole_signs[pole_i] == -1
            ):
                cr_pole_flip = -1.0

            i_update = 0
            # empty `update_on_bin` means "never update"; `decay` stays 0
            update_on_bin_i = (
                update_on_bin[0] if len(update_on_bin) > 0 else -1
            )

            pole = complex(poles[pole_i])
            residue = complex(residues[pole_i])
            state = complex(states[pole_i])

            # `state_prev` lags `state` by one bin, across the call boundary
            # as well: the previous call left both states behind, so the
            # first bin here still reads one that is genuinely two bins old.
            state_prev = complex(states[n_poles + pole_i])

            # A real pole has no implicit complex conjugate (vector-fitting
            # convention): only a pole with imag != 0 stands in for an
            # unstored conjugate partner and needs the doubled injection.
            injection_factor = factor if pole.imag == 0 else two_factor

            decay = 0.0 + 0j
            advance = 0.0 + 0j
            chunk_dt = 0.0
            # The step the previous call took from `state_prev` to `state`;
            # the lag correction of the first bin reaches across it.
            jump_prev = t_state - t_state_prev
            residue_lookback = residue
            bins_since_jump = _STATE_LAG_BINS  # lag factor on the first bins
            for bin_i in range(n_bins):
                if bin_i == update_on_bin_i:
                    chunk_dt = profile_dts[bin_i + 1] - profile_dts[bin_i]
                    decay = np.exp(pole * chunk_dt)
                    if bin_i == 0:
                        t_jump = profile_dts[0] - t_state
                    else:
                        t_jump = profile_dts[bin_i] - profile_dts[bin_i - 1]
                    advance = np.exp(pole * t_jump)
                    bins_since_jump = 0

                    i_update += 1
                    if i_update < len(update_on_bin):
                        update_on_bin_i = update_on_bin[i_update]
                else:
                    t_jump = chunk_dt
                    advance = decay
                if bins_since_jump < _STATE_LAG_BINS:
                    # `state_prev` is referenced two bins back only when the
                    # last two steps were both one bin wide; otherwise reach
                    # across whatever they actually were. The lag is clamped
                    # at zero so the exponent keeps a non-positive real part
                    # and cannot overflow -- a caller handing in a state less
                    # than two bins old has nothing to reach back to, and only
                    # a zero state may do so.
                    lag = t_jump + jump_prev - _STATE_LAG_BINS * bin_dt
                    residue_lookback = (
                        residue * np.exp(pole * lag) if lag > 0.0 else residue
                    )
                    bins_since_jump += 1
                else:
                    residue_lookback = residue
                # Read the state that lags by two bins: the bin-averaged wake
                # starts three half-bins back, so the recursion covers lags of
                # two bins and more. The nearer three lags -- the previous
                # bin, this one and the next -- are added by the solver.
                amp = float(np.real(residue_lookback * state_prev))
                voltage[bin_i] += cr_pole_flip * amp
                state_prev = state
                state = state * advance
                state += cr_pole_flip * profile[bin_i] * injection_factor
                jump_prev = t_jump
            states[pole_i] = state
            states[n_poles + pole_i] = state_prev

        states[-1] = profile_dts[n_bins - 1]
        states[-2] = profile_dts[n_bins - 2]

    @staticmethod
    def music_track(  # NOQA: D102 inherited from `Specials.music_track`
        beam_dt: NumpyArray,
        beam_dE: NumpyArray,
        induced_voltage: NumpyArray,
        parameter_array: NumpyArray,
        alpha: float,
        omega_bar: float,
        const: float,
        coeff1: float,
        coeff2: float,
        coeff3: float,
        coeff4: float,
        time_since_last_track: float,
        multiturn: bool,
    ) -> None:
        if multiturn:
            # Bridge the wake from the previous turn across the rev. gap.
            time_difference_0 = (
                beam_dt[0] + time_since_last_track - parameter_array[2]
            )
            exp_term = np.exp(-alpha * time_difference_0)
            cos_term = np.cos(omega_bar * time_difference_0)
            sin_term = np.sin(omega_bar * time_difference_0)
            product_first = exp_term * (
                (cos_term + coeff1 * sin_term) * parameter_array[0]
                + coeff2 * sin_term * parameter_array[1]
            )
            product_second = exp_term * (
                coeff3 * sin_term * parameter_array[0]
                + (cos_term + coeff4 * sin_term) * parameter_array[1]
            )
        else:
            # Turn 1: no previous-turn wake to bridge.
            product_first = 0.0
            product_second = 0.0

        induced_voltage[0] = const * (0.5 + product_first)
        beam_dE[0] += induced_voltage[0]

        input_first, input_second = _music_recurrence(
            beam_dt,
            beam_dE,
            induced_voltage,
            product_first + 1.0,
            product_second,
            alpha,
            omega_bar,
            const,
            coeff1,
            coeff2,
            coeff3,
            coeff4,
        )
        parameter_array[0] = input_first
        parameter_array[1] = input_second
        parameter_array[2] = beam_dt[len(beam_dt) - 1]


def _music_recurrence(
    beam_dt: NumpyArray,
    beam_dE: NumpyArray,
    induced_voltage: NumpyArray,
    input_first: float,
    input_second: float,
    alpha: float,
    omega_bar: float,
    const: float,
    coeff1: float,
    coeff2: float,
    coeff3: float,
    coeff4: float,
) -> tuple[float, float]:
    """
    Run the MuSiC O(n) recurrence over the sorted macro-particles.

    Updates ``beam_dE`` and ``induced_voltage`` in place (from index 1
    onwards) and returns the carried state for the next turn.

    Parameters
    ----------
    beam_dt
        Macro-particle time coordinates [s], sorted ascending.
    beam_dE
        Macro-particle energy coordinates [eV]; updated in place.
    induced_voltage
        Output induced voltage [V]; written from index 1 onwards.
    input_first
        First component of the carried state at entry.
    input_second
        Second component of the carried state at entry.
    alpha
        Resonator damping ``omega_R / (2 Q)`` [rad/s].
    omega_bar
        Damped resonant angular frequency [rad/s].
    const
        MuSiC prefactor [V].
    coeff1
        Recurrence coefficient.
    coeff2
        Recurrence coefficient.
    coeff3
        Recurrence coefficient.
    coeff4
        Recurrence coefficient.

    Returns
    -------
    input_first
        First component of the carried state after the loop.
    input_second
        Second component of the carried state after the loop.
    """
    for i in range(len(beam_dt) - 1):
        time_difference = beam_dt[i + 1] - beam_dt[i]
        exp_term = np.exp(-alpha * time_difference)
        cos_term = np.cos(omega_bar * time_difference)
        sin_term = np.sin(omega_bar * time_difference)

        product_first = exp_term * (
            (cos_term + coeff1 * sin_term) * input_first
            + coeff2 * sin_term * input_second
        )
        product_second = exp_term * (
            coeff3 * sin_term * input_first
            + (cos_term + coeff4 * sin_term) * input_second
        )

        induced_voltage[i + 1] = const * (0.5 + product_first)
        beam_dE[i + 1] += induced_voltage[i + 1]
        input_first = product_first + 1.0
        input_second = product_second
    return input_first, input_second

    @staticmethod
    def wake_from_twc_fir(
        # read
        profile: NumpyArray,
        grid_index: NumpyArray,
        r_shunt: NumpyArray,
        a_tilde: NumpyArray,
        omega_r: NumpyArray,
        bin_dt: float,
        factor: float,
        # write
        voltage: NumpyArray,
        voltage_threaded: NumpyArray,
    ) -> None:
        """
        Travelling-wave-cavity wake via a phasor FIR recursion.

        Reference implementation; see the ``Specials`` ABC for the full
        description of the algorithm and its lattice-grid convention.

        The bins live on a common equidistant lattice of spacing `bin_dt`
        at positions `grid_index` (integers, strictly increasing). Gaps
        between consecutive bins carry no charge and produce no output;
        the recursion advances across them in closed form, firing each
        taper term's removal at its exact lattice expiry site (the same
        elapsed-time bookkeeping ``wake_from_pole_residue`` uses for its
        ``t_jump``). On a gap-free grid (``grid_index = arange(n_bins)``)
        this reduces to the plain per-bin recursion.

        Parameters
        ----------
        profile
            Beam profile histogram (occupied lattice sites only).
        grid_index
            Lattice site of each profile bin, strictly increasing.
        r_shunt
            Shunt impedance per TWC mode, in [Ohm].
        a_tilde
            Wake support (filling) time per mode, in [s].
        omega_r
            Angular resonant frequency per mode, in [rad/s].
        bin_dt
            Spacing of the underlying equidistant lattice, in [s].
        factor
            To convert `profile` to current per bin [A].
        voltage
            Output voltage, in [V]. Overwritten.
        voltage_threaded
            Cached `voltage` array per thread. For speedup.
        """
        voltage[:] = 0
        voltage_threaded[:, :] = 0

        for mode_i in range(len(r_shunt)):
            _twc_fir_one_mode(
                profile=profile,
                grid_index=grid_index,
                r_shunt=r_shunt[mode_i],
                a_tilde=a_tilde[mode_i],
                omega_r=omega_r[mode_i],
                bin_dt=bin_dt,
                two_factor=2 * factor,
                voltage=voltage,
            )
