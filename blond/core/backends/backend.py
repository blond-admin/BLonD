# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""Global definitions for the capabilities of all backends."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.exceptions import ComplexWarning

from blond.generals.exceptions_ import ArrayCastingError, UnknownBackendMode
from blond.generals.warnings_ import PrecisionWarning

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from types import ModuleType
    from typing import TYPE_CHECKING, Any, Literal, TypeVar

    BackendType = TypeVar("BackendType", bound="type[BackendBaseClass]")

    from cupy.typing import NDArray as CupyArray  # type: ignore
    from numpy.typing import NDArray as NumpyArray

    from blond.generals.typing_ import AnyArray

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "python"
DEFAULT_BITS = "64"

ALL_BACKENDS: dict[str, type[BackendBaseClass]] = {}
# `AVAILABLE_BACKENDS` is provided lazily via the module-level
# `__getattr__` below; see `_probe_available_backends`.


def _register_backend(bd: BackendType) -> BackendType:
    ALL_BACKENDS[bd.__name__] = bd
    return bd


class Specials(ABC):
    """Abstract listing of functions that need implementation for a new backend."""

    @staticmethod
    @abstractmethod  # pragma: no cover
    def get_max_threads() -> int:
        """
        Return the max number of threads this backend's kernels may use.

        Used to size per-thread scratch buffers (e.g. ``voltage_threaded`` in
        ``MultiPoleSparseSolve``). Each backend must report the count from the
        runtime its own kernels will actually use, since numba and the cpp
        backend's libgomp maintain independent thread pools.

        Returns
        -------
        max_threads
            Maximum number of threads this backend's kernels may use.
        """
        raise NotImplementedError(
            "Abstract method `get_max_threads` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def loss_box(  # NOQA: D102
        e_max: float,
        e_min: float,
        t_min: float,
        t_max: float,
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        flags: NumpyArray | CupyArray,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `loss_box` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_single_harmonic(  # NOQA: D102
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: float,
        omega_rf: float,
        phi_rf: float,
        charge: float,
        acceleration_kick: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `kick_single_harmonic` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_multi_harmonic(  # NOQA: D102
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        voltage: NumpyArray,
        omega_rf: NumpyArray,
        phi_rf: NumpyArray,
        charge: float,
        n_rf: int,
        acceleration_kick: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `kick_multi_harmonic` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def sum_1d_array(array: NumpyArray | CupyArray) -> float:
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
        raise NotImplementedError(
            "Abstract method `sum_1d_array` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def dot_product_1d_array(
        array_1: NumpyArray | CupyArray, array_2: NumpyArray | CupyArray
    ) -> float:
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
        raise NotImplementedError(
            "Abstract method `dot_product_1d_array` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_simple(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        eta_0: float,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_simple` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def drift_exact(  # NOQA: D102
        dt: NumpyArray,
        dE: NumpyArray,
        T: float,
        alpha_0: float,
        higher_alpha: NumpyArray,
        beta: float,
        energy: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `drift_exact` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def kick_interpolated(  # NOQA: D102
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

        With the sparse-metadata arguments omitted, `bin_centers` must be
        uniformly spaced; implementations raise `ValueError` otherwise
        (e.g. when handed a gapped, multi-island array such as
        `EquidistantMultiProfile.hist_x` without its metadata). With the
        sparse-metadata arguments given (all six together, typically via
        `EquidistantMultiProfile.sparse_kick_metadata`), particles are
        resolved to their own bucket before interpolation, matching
        `histogram_sparse`'s bucket-resolution semantics.

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
        raise NotImplementedError(
            "Abstract method `kick_interpolated` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def histogram(  # NOQA: D102
        array_read: NumpyArray,
        array_write: NumpyArray,
        start: float,
        stop: float,
    ) -> None:
        raise NotImplementedError(
            "Abstract method `histogram` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def beam_phase(  # NOQA: D102
        hist_x: NumpyArray,
        hist_y: NumpyArray,
        alpha: float,
        omega_rf: float,
        phi_rf: float,
        bin_size: float,
    ) -> float:
        raise NotImplementedError(
            "Abstract method `beam_phase` is not implemented."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def move_flagged_elements_to_end(
        flag: int,
        flags: NumpyArray | CupyArray,  # also purged
        dt: NumpyArray | CupyArray,
        dE: NumpyArray | CupyArray,
        ids: NumpyArray | CupyArray,
    ) -> None:
        """
        Reorder entries where ``flags == flag`` to the array end.

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
        """
        raise NotImplementedError(
            "The backend for `move_flagged_elements_to_end` is missing."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
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
        raise NotImplementedError(
            "The backend for `histogram_sparse` is missing."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def wake_from_pole_residue(
        # read
        profile: NumpyArray | CupyArray,
        profile_dts: NumpyArray | CupyArray,
        poles: NumpyArray | CupyArray,
        residues: NumpyArray | CupyArray,
        is_counterrotating_beam: bool,
        counterrotating_pole_signs: NumpyArray | CupyArray,
        update_on_bin: NumpyArray | CupyArray,
        factor: float,
        # write
        states: NumpyArray | CupyArray,
        voltage: NumpyArray | CupyArray,
        voltage_threaded: NumpyArray | CupyArray,
    ) -> None:
        """
        Apply poles based on the `profile` to generate `voltage`.

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
            Complex state vector, initially ``(0 + 0j)``.
        voltage
            Output voltage, in [V].
        voltage_threaded
            Cached `voltage` array per thread. For speedup.
        """
        raise NotImplementedError(
            "The backend for `wake_from_pole_residue` is missing."
        )

    @staticmethod
    @abstractmethod  # pragma: no cover
    def apply_synchrotron_radiation_and_quantum_excitation_energy_kick(
        beam_dE: NumpyArray | CupyArray,
        energy_lost: float,
        longitudinal_damping_time: float,
        natural_energy_spread: float,
        total_energy: float,
        disable_quantum_excitation: bool = False,
    ) -> None:
        r"""
        Apply synchrotron radiation and quantum excitation energy kicks.

        Updates ``beam_dE`` in place with

        .. math::

            \Delta E \mapsto \left(1 - \frac{2}{\tau}\right)\,\Delta E
                            - U_0
                            + 2 \sigma_\delta \frac{E_0}{\sqrt{\tau}}\,
                              \mathcal{N}(0, 1)

        where the gaussian noise term is omitted when
        ``disable_quantum_excitation`` is ``True``.

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
        raise NotImplementedError(
            "Abstract method "
            "`apply_synchrotron_radiation_and_quantum_excitation_energy_kick` "
            "is not implemented."
        )


class _ModeSwitchHelper:
    """
    Helper to be used in a `with` statement to set the specials temporarily.

    Parameters
    ----------
    backend
        The active backend class.
    mode
        The mode of the specials to be set.
    """

    def __init__(self, backend: BackendBaseClass, mode: str):
        self.backend = backend
        self.mode_org = None
        self.mode_tmp = mode

    def __enter__(self):
        self.mode_org = self.backend.specials_mode
        self.backend.set_specials(mode=self.mode_tmp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.backend.set_specials(mode=self.mode_org)


DEFAULT_FFT_PARALLEL_MIN_SIZE = 2**17  # ~131k samples


class _MaybeParallelFFT:
    """
    FFT façade that opts into multithreading for large CPU transforms.

    Wraps `pyfftw.interfaces.scipy_fft` (CPU) — a `scipy.fft`-compatible
    façade over the FFTW3 library, with plan caching enabled so repeated
    calls at the same size skip FFTW's replanning cost — and passes
    `workers=-1` (all cores) once a transform's size reaches
    `min_size_for_parallel`; below that, thread coordination overhead
    tends to outweigh the FFT cost itself, so transforms stay
    single-threaded. `pyfftw` is a pip-installable wheel on Linux and
    Windows, so this needs no locally-built/maintained FFTW3 bindings
    (see the abandoned `blond/core/backends/cpp/fft.cpp`, which is not
    compiled into the cpp backend and should not be revived for this).
    On CuPy, the transform already runs fully parallel on the GPU and
    `cupy.fft` accepts no `workers` argument, so calls are forwarded
    unchanged.

    Unlike `backend.fft` (`numpy.fft`/`cupy.fft`), this façade has no
    `out=` support: neither `scipy.fft` nor its `pyfftw` façade expose
    it, so buffered/reused output arrays should keep using `backend.fft`
    directly.

    Parameters
    ----------
    fft_module
        `pyfftw.interfaces.scipy_fft` for CPU backends, `cupy.fft` for
        GPU backends.
    is_gpu
        Whether `fft_module` already runs fully parallel on a GPU.
    min_size_for_parallel
        Transform size (in samples) at or above which `workers=-1` is
        passed on CPU. Ignored on GPU.
    """

    def __init__(
        self,
        fft_module: ModuleType,
        is_gpu: bool,
        min_size_for_parallel: int = DEFAULT_FFT_PARALLEL_MIN_SIZE,
    ) -> None:
        self._fft_module = fft_module
        self._is_gpu = is_gpu
        self.min_size_for_parallel = min_size_for_parallel

    def workers_for_size(self, size: int) -> int | None:
        """
        Number of `scipy.fft` workers to use for a transform of `size`.

        Parameters
        ----------
        size
            Length of the transform (i.e. the FFT's `n`).

        Returns
        -------
        int or None
            `None` on GPU (the argument does not apply); `-1` (all
            cores) once `size >= min_size_for_parallel`, else `1`.
        """
        if self._is_gpu:
            return None
        return -1 if size >= self.min_size_for_parallel else 1

    def _with_workers(self, size: int, kwargs: dict) -> dict:
        workers = self.workers_for_size(size)
        if workers is not None:
            kwargs.setdefault("workers", workers)
        return kwargs

    def rfft(self, x: AnyArray, n: int | None = None, **kwargs) -> AnyArray:
        """
        Real-input FFT; see `scipy.fft.rfft`/`cupy.fft.rfft`.

        Parameters
        ----------
        x
            Real-valued input array.
        n
            Transform length. Defaults to `x.shape[-1]`.
        **kwargs
            Forwarded to the underlying `rfft`.

        Returns
        -------
        AnyArray
            The (complex) half-spectrum of `x`.
        """
        size = n if n is not None else x.shape[-1]
        return self._fft_module.rfft(
            x, n=n, **self._with_workers(size, kwargs)
        )

    def irfft(self, x: AnyArray, n: int | None = None, **kwargs) -> AnyArray:
        """
        Inverse of `rfft`; see `scipy.fft.irfft`/`cupy.fft.irfft`.

        Parameters
        ----------
        x
            Half-spectrum, as returned by `rfft`.
        n
            Output length. Defaults to `2 * (x.shape[-1] - 1)`.
        **kwargs
            Forwarded to the underlying `irfft`.

        Returns
        -------
        AnyArray
            The reconstructed real-valued signal.
        """
        size = n if n is not None else 2 * (x.shape[-1] - 1)
        return self._fft_module.irfft(
            x, n=n, **self._with_workers(size, kwargs)
        )

    def fft(self, x: AnyArray, n: int | None = None, **kwargs) -> AnyArray:
        """
        Complex FFT; see `scipy.fft.fft`/`cupy.fft.fft`.

        Parameters
        ----------
        x
            Input array.
        n
            Transform length. Defaults to `x.shape[-1]`.
        **kwargs
            Forwarded to the underlying `fft`.

        Returns
        -------
        AnyArray
            The (complex) full spectrum of `x`.
        """
        size = n if n is not None else x.shape[-1]
        return self._fft_module.fft(x, n=n, **self._with_workers(size, kwargs))

    def ifft(self, x: AnyArray, n: int | None = None, **kwargs) -> AnyArray:
        """
        Inverse complex FFT; see `scipy.fft.ifft`/`cupy.fft.ifft`.

        Parameters
        ----------
        x
            Input spectrum.
        n
            Transform length. Defaults to `x.shape[-1]`.
        **kwargs
            Forwarded to the underlying `ifft`.

        Returns
        -------
        AnyArray
            The reconstructed (complex) signal.
        """
        size = n if n is not None else x.shape[-1]
        return self._fft_module.ifft(
            x, n=n, **self._with_workers(size, kwargs)
        )


class _CachedFftwTransform:
    """
    Buffered real-FFT cache backed by cached `pyfftw.builders` plans.

    Caches one `pyfftw.FFTW` plan per (direction, output size, input
    length), reusing its internal input/output buffers across calls, so
    that repeated calls at the same size -- the common case turn-to-turn
    in a simulation's hot loop, since a `Profile`'s bin count is normally
    fixed for a whole run -- pay neither FFTW's replanning cost nor a new
    array allocation, while still running multithreaded once the
    transform is large enough (`_MaybeParallelFFT.workers_for_size`
    decides the thread count). Unlike `_MaybeParallelFFT`, this keeps the
    zero-allocation behaviour of `out=`-based buffering at every size, not
    just below `min_size_for_parallel`.

    GPU has no equivalent: `cupy.fft` exposes no way to reuse an output
    buffer, only its own internal plan cache (which already amortises
    replanning). So GPU backends should not construct this class and
    should instead point `backend.fft_cached` straight at
    `backend.fft_parallel`.

    Parameters
    ----------
    workers_for_size
        `_MaybeParallelFFT.workers_for_size`, used to pick FFTW's
        `threads` for a newly built plan.
    """

    def __init__(self, workers_for_size: Callable[[int], int | None]) -> None:
        self._workers_for_size = workers_for_size
        self._plans: dict[tuple[str, int, int], Any] = {}

    def _get_plan(self, kind: str, x: AnyArray, n: int) -> tuple[Any, bool]:
        key = (kind, n, x.shape[-1])
        plan = self._plans.get(key)
        if plan is not None:
            return plan, False

        import pyfftw

        builder = (
            pyfftw.builders.rfft if kind == "rfft" else (pyfftw.builders.irfft)
        )
        plan = builder(
            x,
            n=n,
            threads=self._workers_for_size(n),
            planner_effort="FFTW_ESTIMATE",
        )
        self._plans[key] = plan
        return plan, True

    def rfft(self, x: AnyArray, n: int | None = None) -> AnyArray:
        """
        Real-input FFT, buffered and workers-aware; see `scipy.fft.rfft`.

        Parameters
        ----------
        x
            Real-valued input array.
        n
            Transform length. Defaults to `x.shape[-1]`.

        Returns
        -------
        AnyArray
            The (complex) half-spectrum of `x`. The returned array is
            owned by this cache and is overwritten on the next `rfft`
            call of the same `n` and input length -- copy it if it must
            outlive that.
        """
        size = n if n is not None else x.shape[-1]
        plan, freshly_built = self._get_plan("rfft", x, size)
        if not freshly_built:
            plan.input_array[: x.shape[-1]] = x
        return plan()

    def irfft(self, x: AnyArray, n: int | None = None) -> AnyArray:
        """
        Inverse of `rfft`, buffered and workers-aware; see `scipy.fft.irfft`.

        Parameters
        ----------
        x
            Half-spectrum, as returned by `rfft`.
        n
            Output length. Defaults to `2 * (x.shape[-1] - 1)`.

        Returns
        -------
        AnyArray
            The reconstructed real-valued signal. The returned array is
            owned by this cache and is overwritten on the next `irfft`
            call of the same `n` and input length -- copy it if it must
            outlive that.
        """
        size = n if n is not None else 2 * (x.shape[-1] - 1)
        plan, freshly_built = self._get_plan("irfft", x, size)
        if not freshly_built:
            plan.input_array[:] = x
        return plan()


class BackendBaseClass(ABC):
    """
    Base class for a backend.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float64.
    complex_
        Precision type for complex, e.g. complex128.
    specials_mode
        Default mode to load special libraries.
    is_gpu
        Whether the backend is using the GPU.
    verbose
        Enable verbose output.
    """

    # type annotations for MyPy
    float: type[np.float64]
    complex: type[np.complex128]

    def __init__(  # NOQA: PLR0915
        self,
        float_: type[np.float64],
        complex_: type[np.complex128],
        specials_mode: Literal[
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
            "cuda",
        ],
        is_gpu: bool,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose

        self._is_gpu = is_gpu

        if (
            float_ == np.float32 or complex_ == np.complex64
        ):  # pragma: no cover
            warnings.warn(
                "32 Bit backends have been removed, choosing 32 bit float or "
                "64 bit complex will give unpredictable and untested "
                "behaviour.",
                stacklevel=2,
            )

        self.float = float_
        self.complex = complex_

        self.pi = self.float(np.pi)
        self.twopi = self.float(2 * np.pi)
        self.specials_mode = specials_mode
        self.specials: Specials = None  # type: ignore
        self.set_specials(self.specials_mode)

        # Callables that link to e.g. Numpy, Cupy
        self.array: Callable = None  # type: ignore
        self.gradient: Callable = None  # type: ignore
        self.isclose: Callable = None  # type: ignore
        self.empty: Callable = None  # type: ignore
        self.repeat: Callable = None  # type: ignore
        self.linspace: Callable = None  # type: ignore
        self.sinc: Callable = None  # type: ignore
        self.histogram: Callable = None  # type: ignore
        self.histogram2d: Callable = None  # type: ignore
        self.zeros: Callable = None  # type: ignore
        self.ones: Callable = None  # type: ignore
        self.zeros_like: Callable = None  # type: ignore
        self.fft: ModuleType = None  # type: ignore
        self.fft_parallel: _MaybeParallelFFT = None  # type: ignore
        self.fft_cached: _CachedFftwTransform | _MaybeParallelFFT = None  # type: ignore
        self.all: Callable = None  # type: ignore
        self.random: ModuleType = None  # type: ignore
        self.sinc: Callable = None  # type: ignore
        self.isnan: Callable = None  # type: ignore
        self.sum: Callable = None  # type: ignore
        self.sqrt: Callable = None  # type: ignore
        self.interp: Callable = None  # type: ignore
        self.meshgrid: Callable = None  # type: ignore
        self.square: Callable = None  # type: ignore
        self.mean: Callable = None  # type: ignore
        self.arange: Callable = None  # type: ignore
        self.average: Callable = None  # type: ignore
        self.fftconvolve: Callable = None  # type: ignore
        self.min: Callable = None  # type: ignore
        self.max: Callable = None  # type: ignore
        self.dot: Callable = None  # type: ignore
        self.percentile: Callable = None  # type: ignore
        self.cumulative_sum: Callable = None  # type: ignore
        self.array_split: Callable = None  # type: ignore
        self.sign: Callable = None  # type: ignore
        self.sin: Callable = None  # type: ignore
        self.cos: Callable = None  # type: ignore
        self.arctan2: Callable = None  # type: ignore
        self.sinc: Callable = None  # type: ignore
        self.exp: Callable = None  # type: ignore
        self.any: Callable = None  # type: ignore
        self.abs: Callable = None  # type: ignore
        self.convolve: Callable = None  # type: ignore
        self.copy: Callable = None  # type: ignore
        self.ones_like: Callable = None  # type: ignore
        self.add: Callable = None  # type: ignore
        self.default_rng: object = None  # type: ignore
        self.concatenate: Callable = None  # type: ignore
        self.unique: Callable = None  # type: ignore
        self.repeat: Callable = None  # type: ignore
        self.ndarray: type = None  # type: ignore
        self.where: Callable = None  # type: ignore
        self.hstack: type = None  # type: ignore

    def _finalize(self) -> None:
        for attribute, val in self.__dict__.items():
            if val is None:
                raise AttributeError(f"{self.__class__}.{attribute} is None.")

    def autoselect_backend(self) -> None:
        """Set automatically the fastest backend that is available on the computer."""
        order = (
            (Cupy64Bit, "cuda"),
            (Numpy64Bit, "cpp"),
            (Numpy64Bit, "numba"),
            (Numpy64Bit, "python"),
        )
        for backend_, mode_ in order:
            try:
                self.change_backend(new_backend=backend_)
                self.set_specials(mode=mode_)
                return
            except Exception as exc:
                logger.debug(
                    "autoselect: `%s`/`%s` is not available: %r",
                    backend_.__name__,
                    mode_,
                    exc,
                )

    def change_backend(
        self,
        new_backend: type[Numpy64Bit | Cupy64Bit],
    ) -> None:
        """
        Change the backend precision.

        Parameters
        ----------
        new_backend
            One of the available backends.
        """
        if not isinstance(new_backend, type):
            raise TypeError(
                f"`new_backend` must be a {BackendBaseClass.__name__} subclass "
                f"(the class itself, not an instance), got {new_backend!r}."
            )
        if not issubclass(new_backend, BackendBaseClass):
            raise TypeError(
                f"`new_backend` must be a {BackendBaseClass.__name__} subclass, "
                f"got {new_backend!r}."
            )
        if self.__class__ is new_backend:
            # requesting the already active backend must be a no-op
            return
        if self.verbose:
            print(f"Changing backend to `{new_backend.__name__}`")
        _new_backend = new_backend()
        # transfer variables that should be kept when changing backend.
        _new_backend.verbose = self.verbose
        specials_mode_org = self.specials_mode
        self.__dict__ = _new_backend.__dict__
        self.__class__ = _new_backend.__class__
        # If the previous specials mode does not exist on the new backend
        # family (e.g. "cuda" after changing to a CPU backend), keep the
        # new backend's default mode instead. Suppress only that specific
        # case so genuine failures from set_specials still propagate.
        with contextlib.suppress(UnknownBackendMode):
            self.set_specials(specials_mode_org)

    @abstractmethod  # pragma: no cover
    def set_specials(self, mode: Any) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        raise NotImplementedError(
            "Abstract method `set_specials` is not implemented."
        )

    @property
    def is_gpu(self) -> bool:
        """
        Whether the backend is using the GPU.

        Returns
        -------
        is_gpu
            True if the backend is using the GPU, False otherwise.
        """
        return self._is_gpu

    def apply_environment_variables(self) -> None:  # NOQA PLR0915
        """
        Load the environment variables and set up the backend accordingly.

        Notes
        -----
        Following environment variables can be set:

        - `BLOND_BACKEND_MODE` can be 'python', 'cpp', 'cpp_single_core',
          'numba', 'cuda'
        - `BLOND_BACKEND_BITS` can only be '64'
        """
        _backend_mode_env = os.environ.get("BLOND_BACKEND_MODE")
        if _backend_mode_env is not None:
            print(
                f"Using environment variable "
                f"BLOND_BACKEND_MODE = {_backend_mode_env}"
            )
        _backend_mode_raw: str = (
            _backend_mode_env
            if _backend_mode_env is not None
            else DEFAULT_BACKEND
        ).lower()
        _allowed_backend_modes = (
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
            "cuda",
        )
        if _backend_mode_raw in _allowed_backend_modes:
            _backend_mode: Literal[
                "python",
                "cpp",
                "cpp_single_core",
                "numba",
                "cuda",
            ] = _backend_mode_raw  # type: ignore
        else:
            raise ValueError(
                f"The environment variable `BLOND_BACKEND_MODE` "
                f"was set to '{_backend_mode_raw}', but can only be one "
                f"of {_allowed_backend_modes}."
            )

        _backend_bits_raw: str = os.environ.get(
            "BLOND_BACKEND_BITS",
            DEFAULT_BITS,  # default
        )
        _allowed_backend_bits_flag = ("64",)
        if _backend_bits_raw in _allowed_backend_bits_flag:
            _backend_bits: Literal["64",] = _backend_bits_raw  # type: ignore
        else:
            raise ValueError(
                f"The environment variable `BLOND_BACKEND_BITS` "
                f"was set to '{_backend_bits_raw}', but can only be one "
                f"of {_allowed_backend_bits_flag}."
            )

        if _backend_mode == "cuda":
            if _backend_bits == "64":
                self.change_backend(Cupy64Bit)
            else:
                # This statement is not reachable
                # because of `_backend_bits_raw in _allowed_backend_bits_flag`
                # Anyways its beter to write if, elif, else explicitly
                raise ValueError(_backend_bits)  # pragma: no cover
            self.set_specials(mode=_backend_mode)  # type: ignore
        else:
            if _backend_bits == "64":
                self.change_backend(Numpy64Bit)
            else:
                # This statement is not reachable
                # because of `_backend_bits_raw in _allowed_backend_bits_flag`
                # Anyways its beter to write if, elif, else explicitly
                raise ValueError(_backend_bits)  # pragma: no cover
            self.set_specials(mode=_backend_mode)  # type: ignore

    def temporary_specials_mode(self, mode: str):
        """
        Helper to be used in a `with` statement to set the specials temporarily.

        Parameters
        ----------
        mode
            The mode to temporarily switch to.

        Returns
        -------
        mode_switch_helper
            Context manager for temporarily switching modes.

        Examples
        --------
        >>> with backend.temporary_specials_mode("python"):
        ...     print(backend.specials_mode)
        ...     ...
        >>> print(backend.specials_mode)
        """
        return _ModeSwitchHelper(backend=self, mode=mode)

    def _asarray_if_needed(self, arr: AnyArray) -> NumpyArray | CupyArray:
        # Faster to check than cast, so only cast if needed
        if isinstance(arr, self.ndarray):
            return arr

        try:
            gpu_arr = arr.device != "cpu"
        except AttributeError:
            gpu_arr = False

        if gpu_arr:
            arr = arr.get()

        return self.array(arr)

    def _cast_dtype_if_needed(
        self, arr: NumpyArray | CupyArray, dtype: type
    ) -> NumpyArray | CupyArray:
        if arr.dtype != dtype:
            warnings.warn(
                f"Automatically casting dtype from {arr.dtype} to {dtype}",
                stacklevel=3,
                category=PrecisionWarning,
            )
            try:
                # Casting numpy complex array -> float is smooth and
                # includes an automatic ComplexWarning.  Trying to cast
                # a cupy array in the same way raises an exception.
                # Maybe a bug in CuPy?
                # Catch the exception then throw the correct warning.
                arr = arr.astype(dtype)
            # Can be removed some years after 2025.
            except AttributeError as e:  # pragma: no cover
                # Cupy bugfix needed for `cupy-cuda12x<14.0.1`
                if (
                    str(e)
                    == "module 'numpy' has no attribute 'ComplexWarning'"
                ):
                    ComplexWarning(
                        "Casting complex values to real discards the imaginary part"
                    )
                    arr = arr.real.astype(dtype)
                else:  # pragma: no cover
                    raise

        return arr

    def _cast_arr_and_dtype(
        self, arr: AnyArray, dtype: type
    ) -> NumpyArray | CupyArray:
        # Catch likely errors and reraise with slightly friendlier
        # messages.  Raise from the original exception to aid
        # debugging.
        #
        # ValueError is raised by backend.array(arr) if input is ragged.
        # TypeError is raised by arr.astype(backend.[type]) if input
        # cannot be coerced to the new type (e.g. str -> float)

        try:
            new_arr = self._asarray_if_needed(arr)
        except ValueError as exc:
            raise ArrayCastingError(
                f"Unable to convert input data {arr} to array."
            ) from exc

        try:
            new_arr = self._cast_dtype_if_needed(new_arr, dtype)
        except (TypeError, ValueError) as exc:
            raise ArrayCastingError(
                "Unable to automatically cast dtype of input data from "
                f"{new_arr.dtype} to {dtype}."
            ) from exc

        return new_arr

    def cast_arr_float_if_needed(
        self, arr: AnyArray
    ) -> NumpyArray | CupyArray:
        """
        Convert input to backend.array with ``dtype=backend.float``.

        Uses isinstance and dtype checks to only modify the object if
        needed, which is faster and avoids breaking references.  If the
        reference is required to change, `backend.array` should be
        called directly.

        Parameters
        ----------
        arr
            The object that should be returned as an array.

        Returns
        -------
        NumpyArray | CupyArray
            The modified (if needed) array.
        """
        return self._cast_arr_and_dtype(arr, self.float)

    def cast_arr_complex_if_needed(
        self, arr: AnyArray
    ) -> NumpyArray | CupyArray:
        """
        Convert input to backend.array with ``dtype=backend.complex``.

        Uses isinstance and dtype checks to only modify the object if
        needed, which is faster and avoids breaking references.  If the
        reference is required to change, `backend.array` should be
        called directly.

        Parameters
        ----------
        arr
            The object that should be returned as an array.

        Returns
        -------
        NumpyArray | CupyArray
            The modified (if needed) array.
        """
        return self._cast_arr_and_dtype(arr, self.complex)


class NumpyBackend(BackendBaseClass):
    """
    Base class for Numpy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float64.
    complex_
        Precision type for complex, e.g. complex128.
    """

    def __init__(  # NOQA: PLR0915
        self,
        float_: type[np.float64],
        complex_: type[np.complex128],
    ) -> None:
        super().__init__(
            float_,
            complex_,
            specials_mode="python",
            is_gpu=False,
        )
        import pyfftw.interfaces.cache as _pyfftw_cache
        import pyfftw.interfaces.scipy_fft as _pyfftw_scipy_fft
        from scipy.signal import fftconvolve

        _pyfftw_cache.enable()

        self.array = np.array
        self.gradient = np.gradient
        self.isclose = np.isclose
        self.empty = np.empty
        self.repeat = np.repeat
        self.linspace = np.linspace
        self.sinc = np.sinc
        self.histogram = np.histogram
        self.histogram2d = np.histogram2d
        self.zeros = np.zeros
        self.ones = np.ones
        self.zeros_like = np.zeros_like
        self.fft = np.fft
        self.fft_parallel = _MaybeParallelFFT(_pyfftw_scipy_fft, is_gpu=False)
        self.fft_cached = _CachedFftwTransform(
            self.fft_parallel.workers_for_size
        )
        self.all = np.all
        self.random = np.random
        self.sinc = np.sinc
        self.isnan = np.isnan
        self.sum = np.sum
        self.sqrt = np.sqrt
        self.interp = np.interp
        self.meshgrid = np.meshgrid
        self.square = np.square
        self.mean = np.mean
        self.arange = np.arange
        self.average = np.average
        self.fftconvolve = fftconvolve
        self.min = np.min
        self.max = np.max
        self.dot = np.dot
        self.percentile = np.percentile
        try:  # pragma: no cover
            self.cumulative_sum = np.cumulative_sum
        except AttributeError:  # pragma: no cover
            self.cumulative_sum = np.cumsum
        self.array_split = np.array_split
        self.sign = np.sign
        self.sin = np.sin
        self.cos = np.cos
        self.arctan2 = np.arctan2
        self.sinc = np.sinc
        self.exp = np.exp
        self.any = np.any
        self.abs = np.abs
        self.convolve = np.convolve
        self.copy = np.copy
        self.ones_like = np.ones_like
        self.add = np.add
        self.default_rng = np.random.default_rng
        self.concatenate = np.concatenate
        self.unique = np.unique
        self.repeat = np.repeat
        self.ndarray = np.ndarray
        self.where = np.where
        self.hstack = np.hstack

        self._finalize()

    def set_specials(
        self,
        mode: Literal[
            "python",
            "cpp",
            "cpp_single_core",
            "numba",
        ],
    ) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        onchange = self.specials_mode != mode

        if mode == "python":
            from blond.core.backends.python.callables import PythonSpecials

            self.specials = PythonSpecials()
            self.specials_mode = mode
        elif mode == "cpp":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(self.float, parallel=True)
            self.specials_mode = mode
        elif mode == "cpp_single_core":
            from blond.core.backends.cpp.callables import reload_cpp_backend

            self.specials = reload_cpp_backend(self.float, parallel=False)
            self.specials_mode = mode
        elif mode == "numba":
            from blond.core.backends.numba.callables import (
                NumbaSpecials,
            )

            self.specials = NumbaSpecials()
            self.specials_mode = mode
        else:
            raise UnknownBackendMode(
                f"Unknown specials mode {mode!r} for {type(self).__name__}."
            )
        if self.verbose and onchange:
            print(f"Set special to `{mode}`")


@_register_backend
class Numpy64Bit(NumpyBackend):
    """Numpy backend with 64 bit precision."""

    def __init__(
        self,
    ) -> None:
        super().__init__(
            np.float64,
            np.complex128,
        )


class CupyBackend(BackendBaseClass):
    """
    Base class for Cupy based backends.

    Parameters
    ----------
    float_
        Precision type for float, e.g. float64.
    complex_
        Precision type for complex, e.g. complex128.
    """

    def __init__(  # NOQA: PLR0915
        self,
        float_: type[np.float64],
        complex_: type[np.complex128],
    ) -> None:
        super().__init__(
            float_,
            complex_,
            specials_mode="cuda",  # no other backend implemented at the moment
            is_gpu=True,
        )
        import cupy as cp  # type: ignore # import only if needed, which is not always the case

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message="cupyx.jit.rawkernel is experimental. The interface can change in the future.",
            )
            from cupyx.scipy.signal import fftconvolve

        self.array = cp.array
        self.gradient = cp.gradient
        self.isclose = cp.isclose
        self.empty = cp.empty
        self.repeat = cp.repeat
        self.linspace = cp.linspace
        self.sinc = cp.sinc
        self.histogram = cp.histogram
        self.histogram2d = cp.histogram2d
        self.zeros = cp.zeros
        self.ones = cp.ones
        self.zeros_like = cp.zeros_like
        self.fft = cp.fft
        self.fft_parallel = _MaybeParallelFFT(cp.fft, is_gpu=True)
        # cupy.fft has no way to reuse an output buffer (only its own
        # internal plan cache), so there is no cached/buffered variant
        # to offer here; callers get the same behaviour either way.
        self.fft_cached = self.fft_parallel
        self.all = cp.all
        self.random = cp.random
        self.sinc = cp.sinc
        self.isnan = cp.isnan
        self.sum = cp.sum
        self.sqrt = cp.sqrt
        self.interp = cp.interp
        self.meshgrid = cp.meshgrid
        self.square = cp.square
        self.mean = cp.mean
        self.arange = cp.arange
        self.average = cp.average
        self.fftconvolve = fftconvolve
        self.min = cp.min
        self.max = cp.max
        self.dot = cp.dot
        self.percentile = cp.percentile
        self.cumulative_sum = cp.cumsum
        self.array_split = cp.array_split
        self.sign = cp.sign
        self.sin = cp.sin
        self.cos = cp.cos
        self.arctan2 = cp.arctan2
        self.sinc = cp.sinc
        self.exp = cp.exp
        self.any = cp.any
        self.abs = cp.abs
        self.convolve = cp.convolve
        self.copy = cp.copy
        self.ones_like = cp.ones_like
        self.add = cp.add
        self.default_rng = cp.random.default_rng
        self.concatenate = cp.concatenate
        self.unique = cp.unique
        self.repeat = cp.repeat
        self.ndarray = cp.ndarray
        self.where = cp.where
        self.hstack = cp.hstack

        from blond.core.backends.cuda.callables import CudaSpecials

        self.specials = CudaSpecials()

        self._finalize()

    def set_specials(self, mode: Literal["cuda"]) -> None:
        """
        Set the special compiled functions.

        Parameters
        ----------
        mode
            One of the available backend modes.
        """
        if mode == "cuda":
            from blond.core.backends.cuda.callables import CudaSpecials

            self.specials = CudaSpecials()
        else:
            raise UnknownBackendMode(
                f"Unknown specials mode {mode!r} for {type(self).__name__}."
            )
        if self.verbose:
            print(f"Set special to `{mode}`")


@_register_backend
class Cupy64Bit(CupyBackend):
    """Cupy backend with 64 bit precision."""

    def __init__(self) -> None:
        super().__init__(
            np.float64,
            np.complex128,
        )


def _probe_available_backends() -> dict[str, type[BackendBaseClass]]:
    """
    Probe which of the registered backends can be instantiated.

    Returns
    -------
    available_backends
        Mapping from backend name to backend class.
    """
    available: dict[str, type[BackendBaseClass]] = {}
    for k, v in ALL_BACKENDS.items():
        try:
            v()
        # Skip on any exception, we only care that it's not available.
        except Exception as exc:  # pragma: no cover
            logger.debug("Backend `%s` is not available: %r", k, exc)
        else:
            available[k] = v
    return available


def __getattr__(name: str):
    """
    Provide lazy module attributes (PEP 562).

    Probing `AVAILABLE_BACKENDS` instantiates every registered backend,
    which for CUDA queries the GPU and may even trigger a compilation.
    This must not happen as a side effect of importing this module.

    Parameters
    ----------
    name
        Name of the requested module attribute.

    Returns
    -------
    attribute
        The lazily created module attribute.
    """
    if name == "AVAILABLE_BACKENDS":
        available = _probe_available_backends()
        globals()["AVAILABLE_BACKENDS"] = available
        return available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


default = Numpy64Bit  # use .change_backend(...) to change it anywhere
backend: Numpy64Bit | Cupy64Bit = default()
backend.apply_environment_variables()
# verbose only after the initial setup, so that importing blond stays
# quiet, but later backend changes are reported to the user
backend.verbose = True
