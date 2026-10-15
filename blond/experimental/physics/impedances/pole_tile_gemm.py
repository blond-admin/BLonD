# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
The far field of a pole-residue wake as tiled GEMMs (experimental).

`Specials.wake_from_pole_residue` recurses bin by bin, one state per pole.
`PoleTileGemm` evaluates the same far field with the poles as a diagonal
linear state-space model, in tiles of K contiguous bins:

    Z_t     = h_t . F                     tile t's own sum, all poles
    Y_{t+1} = A_t Y_t + G_t Z_t + ...     scan over the tiles
    V_t     = Re(Y_t . E) + h_t . T       read-outs of tile t

with h_t the tile's charges (and a halo, the previous bin if adjacent),
d_p = exp(p bin_dt), F = inj d^(K - 1 - s), E = r d^k, and T the wake
summed over the poles at lags below a tile. The sum over the poles is the
inner dimension of both GEMMs, so it never goes through atomics; on a GPU
cuBLAS runs them on the FP64 tensor cores. The scan over the tiles is the
only sequential part; on a GPU it is two-level (groups of tiles scanned in
parallel, then stitched), so it takes ~2 sqrt(2 n_tiles) steps.

Memory: [tiles x poles] arrays per slab of tiles, bounded by
``max_slab_bytes``; the grid is processed in as many slabs as needed.

`MultiPoleGemmSolve` is `MultiPoleSparseSolve` with this far field.
Requirements: bins on a whole-bin grid (checked by `MultiPoleSparseSolve`),
and contiguous runs of bins that, except the last, are a multiple of the
tile length (16 to 128 bins; true for an `EquidistantMultiProfile` whose
`bins_per_profile` is a multiple of 16).
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from blond.core.backends.backend import backend
from blond.generals.cupy_.no_cupy_import import copy_to_cpu, is_cupy_array
from blond.physics.impedances.solvers import MultiPoleSparseSolve

if TYPE_CHECKING:  # pragma: no cover
    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.core.beam.base import BeamBaseClass

# Tile lengths tried, longest first.
_TILE_LENGTHS = (128, 64, 32, 16)

_SCAN_SOURCE = r"""
// The scan over the tiles of a slab, two-level: the slab's tiles are cut
// into groups; `pole_scan_local` scans every group from zero and keeps the
// group's propagator, `pole_scan_groups` stitches the groups, sequential
// over them, and `pole_scan_fix` adds each group's true start state.
// Arrays [tiles][poles] are pole-fastest; complex values are planar.
// The step after tile t, of type `type_ids[t]` (-1: no next tile):
//   Y_{t+1} = A Y_t + G Z_t + inj h_last LB   (+ carried after tile 0)

__device__ inline void cmul(double ar, double ai, double br, double bi,
                            double &cr, double &ci) {
  cr = ar * br - ai * bi;
  ci = ar * bi + ai * br;
}

extern "C" __global__ void pole_scan_local(
    const double *z_re, const double *z_im, const int *type_ids,
    const double *last_charge, const double *a_re, const double *a_im,
    const double *g_re, const double *g_im, const double *lb_re,
    const double *lb_im, const double *injection, const double *carry_re,
    const double *carry_im, const int carry_tile, const int n_tiles,
    const int n_poles, const int group_length, double *y_re, double *y_im,
    double *end_re, double *end_im, double *prop_re, double *prop_im) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int g = blockIdx.y;
  if (p >= n_poles)
    return;
  const int t0 = g * group_length;
  const int t1 = min(t0 + group_length, n_tiles);
  double yr = 0, yi = 0, pr = 1, pi = 0;
  const double inj = injection[p];
  for (int t = t0; t < t1; ++t) {
    const size_t i = (size_t)t * n_poles + p;
    y_re[i] = yr;
    y_im[i] = yi;
    const int type = type_ids[t];
    if (type < 0)
      continue;  // the last tile: no step after it
    const size_t k = (size_t)type * n_poles + p;
    double ar = a_re[k], ai = a_im[k], nr, ni, br, bi;
    cmul(ar, ai, yr, yi, nr, ni);
    cmul(g_re[k], g_im[k], z_re[i], z_im[i], br, bi);
    nr += br + inj * last_charge[t] * lb_re[k];
    ni += bi + inj * last_charge[t] * lb_im[k];
    if (t == carry_tile) {
      nr += carry_re[p];
      ni += carry_im[p];
    }
    yr = nr;
    yi = ni;
    double qr, qi;
    cmul(ar, ai, pr, pi, qr, qi);
    pr = qr;
    pi = qi;
  }
  const size_t j = (size_t)g * n_poles + p;
  end_re[j] = yr;
  end_im[j] = yi;
  prop_re[j] = pr;
  prop_im[j] = pi;
}

extern "C" __global__ void pole_scan_groups(
    const double *end_re, const double *end_im, const double *prop_re,
    const double *prop_im, const int n_groups, const int n_poles,
    double *start_re, double *start_im, double *state_re, double *state_im) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_poles)
    return;
  double sr = state_re[p], si = state_im[p];
  for (int g = 0; g < n_groups; ++g) {
    const size_t j = (size_t)g * n_poles + p;
    start_re[j] = sr;
    start_im[j] = si;
    double nr, ni;
    cmul(prop_re[j], prop_im[j], sr, si, nr, ni);
    sr = nr + end_re[j];
    si = ni + end_im[j];
  }
  state_re[p] = sr;  // the state at the next slab's first tile
  state_im[p] = si;
}

extern "C" __global__ void pole_scan_fix(
    const int *type_ids, const double *a_re, const double *a_im,
    const double *start_re, const double *start_im, const int n_tiles,
    const int n_poles, const int group_length, double *y_re, double *y_im) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int g = blockIdx.y;
  if (p >= n_poles)
    return;
  const int t0 = g * group_length;
  const int t1 = min(t0 + group_length, n_tiles);
  const size_t j = (size_t)g * n_poles + p;
  double cr = start_re[j], ci = start_im[j];
  for (int t = t0; t < t1; ++t) {
    const size_t i = (size_t)t * n_poles + p;
    y_re[i] += cr;
    y_im[i] += ci;
    const int type = type_ids[t];
    if (type < 0)
      break;
    const size_t k = (size_t)type * n_poles + p;
    double nr, ni;
    cmul(a_re[k], a_im[k], cr, ci, nr, ni);
    cr = nr;
    ci = ni;
  }
}
"""


@functools.cache
def _scan_kernels():  # pragma: no cover (GPU only)
    import cupy as cp  # noqa: PLC0415  (GPU only)

    module = cp.RawModule(code=_SCAN_SOURCE)
    return (
        module.get_function("pole_scan_local"),
        module.get_function("pole_scan_groups"),
        module.get_function("pole_scan_fix"),
    )


@njit(parallel=True, cache=False)
def _scan_cpu(  # noqa: PLR0913
    z_re,
    z_im,
    type_ids,
    last_charge,
    a_re,
    a_im,
    g_re,
    g_im,
    lb_re,
    lb_im,
    injection,
    carry_re,
    carry_im,
    carry_tile,
    state_re,
    state_im,
    y_re,
    y_im,
):  # pragma: no cover (jitted)
    """The scan over a slab's tiles on the CPU, sequential per pole."""
    n_tiles, n_poles = z_re.shape
    for p in prange(n_poles):
        yr, yi = state_re[p], state_im[p]
        for t in range(n_tiles):
            y_re[t, p] = yr
            y_im[t, p] = yi
            kind = type_ids[t]
            if kind < 0:
                continue
            ar, ai = a_re[kind, p], a_im[kind, p]
            nr = ar * yr - ai * yi
            ni = ar * yi + ai * yr
            gr, gi = g_re[kind, p], g_im[kind, p]
            nr += gr * z_re[t, p] - gi * z_im[t, p]
            ni += gr * z_im[t, p] + gi * z_re[t, p]
            nr += injection[p] * last_charge[t] * lb_re[kind, p]
            ni += injection[p] * last_charge[t] * lb_im[kind, p]
            if t == carry_tile:
                nr += carry_re[p]
                ni += carry_im[p]
            yr, yi = nr, ni
        state_re[p], state_im[p] = yr, yi


class PoleTileGemm:
    """
    The far field of a pole-residue wake as tiled GEMMs, for a fixed grid.

    Parameters
    ----------
    bin_offsets
        Position of each bin in whole bins from the first bin, increasing.
    poles
        Complex poles, in [rad/s].
    residues
        Complex residues, scaled to the bin-averaged far field.
    counterrotating_pole_signs
        Per pole, -1 if the sign is flipped for a counter-rotating beam.
    bin_dt
        Bin width, in [s].
    max_slab_bytes
        Memory budget of the per-slab [tiles x poles] arrays, in bytes.

    Raises
    ------
    ValueError
        If the runs of contiguous bins (but the last) have no common tile
        length of 16 bins or more.
    """

    def __init__(
        self,
        bin_offsets: NumpyArray,
        poles: NumpyArray | CupyArray,
        residues: NumpyArray | CupyArray,
        counterrotating_pole_signs: NumpyArray | CupyArray,
        bin_dt: float,
        max_slab_bytes: int = 2**31,
    ) -> None:
        offsets = np.asarray(bin_offsets, dtype=np.int64)
        offsets = offsets - offsets[0]
        n_bins = len(offsets)
        breaks = np.flatnonzero(np.diff(offsets) != 1) + 1
        run_starts = np.concatenate(([0], breaks))
        run_lengths = np.diff(np.concatenate((run_starts, [n_bins])))
        tile_length = next(
            (k for k in _TILE_LENGTHS if np.all(run_lengths[:-1] % k == 0)),
            None,
        )
        if tile_length is None:
            raise ValueError(
                "PoleTileGemm: the runs of contiguous bins, but the last, "
                f"must share a tile length of {_TILE_LENGTHS[-1]} bins or "
                f"more; their lengths are {sorted(set(run_lengths[:-1]))}."
            )
        k = tile_length
        first = np.concatenate(
            [np.arange(s, s + n, k) for s, n in zip(run_starts, run_lengths)]
        )
        n_tiles = len(first)
        self.n_bins, self.n_tiles, self.tile_length = n_bins, n_tiles, k
        self.bin_dt = bin_dt
        self._last_offset = int(offsets[-1])
        self._tile_first = first
        self._last_tile_length = n_bins - int(first[-1])
        starts = offsets[first]
        self._second_start = int(starts[1]) if n_tiles > 1 else 0
        # charges per tile: halo, then the tile's bins; n_bins is a zero
        halo = np.full(n_tiles, n_bins)
        adjacent = offsets[first[1:] - 1] + 1 == starts[1:]
        halo[1:][adjacent] = first[1:][adjacent] - 1
        columns = first[:, None] + np.arange(k)[None, :]
        columns[columns >= np.append(first[1:], n_bins)[:, None]] = n_bins
        self._gather = backend.array(
            np.concatenate((halo[:, None], columns), axis=1), dtype=np.int64
        )
        last_bin = np.minimum(first + k - 1, n_bins)
        last_bin[-1] = n_bins  # no step after the last tile
        self._last_bin = backend.array(last_bin, dtype=np.int64)
        # steps between tiles: their gap beyond a tile, as types
        gaps = starts[1:] - starts[:-1] - k
        type_gaps, type_ids = np.unique(gaps, return_inverse=True)
        self._type_ids = np.append(type_ids, -1).astype(np.int32)

        poles = backend.array(poles, dtype=backend.complex)
        residues = backend.array(residues, dtype=backend.complex)
        self._signs = backend.array(counterrotating_pole_signs)
        self._pole_dt = poles * bin_dt
        self._pair = 2.0 - (poles.imag == 0)
        self._residues = residues
        self._powers = backend.exp(
            backend.arange(k + 1)[:, None] * self._pole_dt[None, :]
        )  # d^m, m = 0..K
        # per step type: decay over the step, of Z over the gap, and of the
        # tile's last bin over the gap but one (none without a gap)
        type_gaps = backend.array(type_gaps.astype(np.float64))
        step = backend.exp((type_gaps[:, None] + k) * self._pole_dt[None, :])
        own = backend.exp(type_gaps[:, None] * self._pole_dt[None, :])
        last = backend.where(
            type_gaps[:, None] > 0,
            backend.exp((type_gaps[:, None] - 1) * self._pole_dt[None, :]),
            0.0,
        )
        self._step = (step.real.copy(), step.imag.copy())
        self._own = (own.real.copy(), own.imag.copy())
        self._last = (last.real.copy(), last.imag.copy())
        # the wake summed over all poles at lags below a tile; the flips of
        # injection and read-out cancel
        wake = (
            (self._powers[:k] * (residues * self._pair)[None, :])
            .sum(axis=1)
            .real
        )
        lag = np.arange(k)[None, :] - np.arange(k + 1)[:, None] - 1
        self._toeplitz = backend.array(
            np.where(lag >= 0, copy_to_cpu(wake)[np.clip(lag, 0, k - 1)], 0.0),
            dtype=backend.float,
        )
        self._by_rotation: dict[bool, tuple] = {}
        bytes_per_tile = len(poles) * 8 * 8 + (k + 1) * 8
        self._slab_tiles = max(
            1, min(n_tiles, max_slab_bytes // bytes_per_tile)
        )

    def _matrices(self, is_counterrotating_beam: bool) -> tuple:
        """Injection, F and E for one direction of the beam, cached."""
        if is_counterrotating_beam not in self._by_rotation:
            k = self.tile_length
            flip = backend.where(
                (self._signs == -1) & is_counterrotating_beam, -1.0, 1.0
            )
            injection = flip * self._pair
            readout = flip * self._residues
            source = backend.zeros(
                (k + 1, len(injection)), dtype=backend.complex
            )
            source[:k] = injection[None, :] * self._powers[k - 1 :: -1]
            reads = (readout[None, :] * self._powers[:k]).T
            self._by_rotation[is_counterrotating_beam] = (
                injection,
                readout,
                source.real.copy(),
                source.imag.copy(),
                reads.real.copy(),
                reads.imag.copy(),
            )
        return self._by_rotation[is_counterrotating_beam]

    def far_field(
        self,
        profile: NumpyArray | CupyArray,
        carried_charge: NumpyArray | CupyArray,
        carried_is_counterrotating: bool,
        state_lag_dt: float,
        carried_lag_dt: float,
        is_counterrotating_beam: bool,
        factor: float,
        states: NumpyArray | CupyArray,
    ) -> NumpyArray | CupyArray:
        """
        ``Specials.wake_from_pole_residue`` on this grid; see its contract.

        Parameters
        ----------
        profile
            Beam profile histogram, one value per bin.
        carried_charge
            Length 1: charge of the previous call's last bin.
        carried_is_counterrotating
            Whether the call of the carried charge was counter-rotating.
        state_lag_dt
            How far before the first read-out the states are referenced,
            in [s].
        carried_lag_dt
            How long before the first bin the carried charge was emitted,
            in [s].
        is_counterrotating_beam
            Whether the beam is counter-rotating.
        factor
            To convert `profile` to charge per bin.
        states
            Complex state per pole; read and overwritten with the state one
            bin before the last bin.

        Returns
        -------
        voltage
            The far-field voltage per bin, in [V].
        """
        k, n_tiles = self.tile_length, self.n_tiles
        injection, readout, f_re, f_im, e_re, e_im = self._matrices(
            is_counterrotating_beam
        )
        pole_dt = self._pole_dt
        charge = backend.zeros(self.n_bins + 1, dtype=backend.float)
        charge[: self.n_bins] = profile * factor
        carried_flip = backend.where(
            (self._signs == -1) & carried_is_counterrotating, -1.0, 1.0
        )
        # the carried charge, at -1, and the incoming state, at -2
        carried = (
            carried_flip
            * self._pair
            * carried_charge[0]
            * backend.exp(pole_dt * (carried_lag_dt / self.bin_dt - 1.0))
        )
        state = states * backend.exp(pole_dt * (state_lag_dt / self.bin_dt))
        carry_into_tile_1 = carried * backend.exp(
            pole_dt * (self._second_start - 1)
        )
        state_re = state.real.copy()
        state_im = state.imag.copy()

        voltage = backend.zeros(self.n_bins, dtype=backend.float)
        for t0 in range(0, n_tiles, self._slab_tiles):
            t1 = min(t0 + self._slab_tiles, n_tiles)
            charges = charge[self._gather[t0:t1]]
            z_re, z_im = charges @ f_re, charges @ f_im
            y_re, y_im = self._scan(
                z_re,
                z_im,
                t0,
                t1,
                charge[self._last_bin[t0:t1]],
                injection,
                carry_into_tile_1,
                state_re,
                state_im,
            )
            reads = y_re @ e_re - y_im @ e_im + charges @ self._toeplitz
            if t0 == 0:  # the carried charge in tile 0, from read-out 1 on
                carried_reads = (carried @ (e_re + 1j * e_im)).real
                reads[0, 1:] += carried_reads[: k - 1]
            first = int(self._tile_first[t0])
            stop = int(self._tile_first[t1]) if t1 < n_tiles else self.n_bins
            voltage[first:stop] = reads.reshape(-1)[: stop - first]
            if t1 == n_tiles:
                last_charges = charges[-1]
                last_state = y_re[-1] + 1j * y_im[-1]
        # hand over: one bin before the last bin, every bin but the last in
        length = self._last_tile_length
        own = last_charges[:length] @ (
            injection[None, :] * self._powers[length - 1 :: -1][:length]
        )
        handover = self._powers[length] * last_state + own
        if n_tiles == 1:
            handover += carried * backend.exp(pole_dt * self._last_offset)
        states[:] = handover
        return voltage

    def _scan(  # noqa: PLR0913
        self,
        z_re,
        z_im,
        t0,
        t1,
        last_charge,
        injection,
        carry_into_tile_1,
        state_re,
        state_im,
    ):
        """Y over tiles [t0, t1); advances `state_*` to tile t1."""
        n_tiles, n_poles = t1 - t0, z_re.shape[1]
        type_ids = backend.array(self._type_ids[t0:t1], dtype=np.int32)
        carry_tile = 0 - t0 if t0 == 0 else -1  # step after global tile 0
        carry_re = carry_into_tile_1.real.copy()
        carry_im = carry_into_tile_1.imag.copy()
        y_re = backend.empty((n_tiles, n_poles), dtype=backend.float)
        y_im = backend.empty((n_tiles, n_poles), dtype=backend.float)
        tables = (*self._step, *self._own, *self._last)
        if not is_cupy_array(z_re):
            _scan_cpu(
                z_re,
                z_im,
                type_ids,
                last_charge,
                *tables,
                injection,
                carry_re,
                carry_im,
                carry_tile,
                state_re,
                state_im,
                y_re,
                y_im,
            )
            return y_re, y_im
        local, groups, fix = _scan_kernels()  # pragma: no cover (GPU only)
        n_groups = max(1, min(n_tiles, round(math.sqrt(2 * n_tiles))))
        group_length = -(-n_tiles // n_groups)
        n_groups = -(-n_tiles // group_length)
        ends = [backend.empty((n_groups, n_poles)) for _ in range(6)]
        threads = 128
        grid = (-(-n_poles // threads), n_groups)
        local(
            grid,
            (threads,),
            (
                z_re,
                z_im,
                type_ids,
                last_charge,
                *tables,
                injection,
                carry_re,
                carry_im,
                np.int32(carry_tile),
                np.int32(n_tiles),
                np.int32(n_poles),
                np.int32(group_length),
                y_re,
                y_im,
                *ends[:4],
            ),
        )
        groups(
            (-(-n_poles // threads),),
            (threads,),
            (
                *ends[:4],
                np.int32(n_groups),
                np.int32(n_poles),
                *ends[4:],
                state_re,
                state_im,
            ),
        )
        fix(
            grid,
            (threads,),
            (
                type_ids,
                *self._step,
                *ends[4:],
                np.int32(n_tiles),
                np.int32(n_poles),
                np.int32(group_length),
                y_re,
                y_im,
            ),
        )
        return y_re, y_im


class MultiPoleGemmSolve(MultiPoleSparseSolve):
    """
    `MultiPoleSparseSolve` with the far field as tiled GEMMs (experimental).

    Same model and results as `MultiPoleSparseSolve`; the far field is
    evaluated by `PoleTileGemm` instead of `Specials.wake_from_pole_residue`.
    Faster for many poles on GPUs with FP64 tensor cores, at the cost of
    [tiles x poles] working arrays, bounded by `max_slab_bytes`.

    Parameters
    ----------
    far_field_tolerance
        See `MultiPoleSparseSolve`.
    max_slab_bytes
        Memory budget of the per-slab working arrays, in bytes.
    """

    def __init__(
        self, far_field_tolerance: float = 1e-12, max_slab_bytes: int = 2**31
    ) -> None:
        super().__init__(far_field_tolerance=far_field_tolerance)
        self._max_slab_bytes = max_slab_bytes
        self._gemm: PoleTileGemm | None = None

    def _far_field_voltage(
        self,
        beam: BeamBaseClass,
        hist_x: NumpyArray | CupyArray,
        hist_y: NumpyArray | CupyArray,
        factor: float,
        carried_lag_dt: float,
    ) -> NumpyArray | CupyArray:
        """See `MultiPoleSparseSolve._far_field_voltage`."""
        far_field, carry = self._far_field, self._carry
        bin_dt = self._model.bin_dt
        if len(far_field.poles) == 0:
            return backend.zeros(len(hist_y), dtype=backend.float)
        if self._gemm is None:
            offsets = np.rint(copy_to_cpu(hist_x - hist_x[0]) / bin_dt)
            self._gemm = PoleTileGemm(
                bin_offsets=offsets.astype(np.int64),
                poles=far_field.poles,
                residues=far_field.residues,
                counterrotating_pole_signs=far_field.pole_signs,
                bin_dt=bin_dt,
                max_slab_bytes=self._max_slab_bytes,
            )
        return self._gemm.far_field(
            profile=hist_y,
            carried_charge=carry.charge,
            carried_is_counterrotating=carry.is_counter_rotating,
            state_lag_dt=carried_lag_dt - bin_dt,
            carried_lag_dt=carried_lag_dt,
            is_counterrotating_beam=beam.is_counter_rotating,
            factor=factor,
            states=self._states,
        )
