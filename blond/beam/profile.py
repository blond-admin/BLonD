# coding: utf-8
# Copyright 2017 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
**Module to compute the beam profile through slices**

:Authors: **Danilo Quartullo**, **Alexandre Lasheen**,
          **Juan F. Esteban Mueller**
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage

from ..toolbox import filters_and_fitting as ffroutines
from ..utils import bmath as bm
from ..utils.legacy_support import handle_legacy_kwargs

if TYPE_CHECKING:
    from typing import Callable, Optional, Tuple

    from numpy.typing import NDArray as NumpyArray

    from ..utils.types import DeviceType
    from .beam import Beam
    from ..input_parameters.rf_parameters import RFStation
    from ..utils.types import (
        FilterExtraOptionsType,
        CutUnitType,
        FitOptionTypes,
        FilterMethodType,
        BeamProfileDerivativeModes,
    )


class CutOptions:
    r"""
    This class groups all the parameters necessary to slice the phase space
    distribution according to the time axis, apart from the array collecting
    the profile which is defined in the constructor of the class Profile below.

    Parameters
    ----------
    cut_left : float
        Left edge of the slicing (optional). A default value will be set if
        no value is given.
    cut_right : float
        Right edge of the slicing (optional). A default value will be set
        if no value is given.
    n_slices : int
        Optional input parameters, corresponding to the number of
        :math:`\sigma_{RMS}` of the Beam to slice (this will overwrite
        any input of cut_left and cut_right).
    n_sigma : float
        defines the left and right extremes of the profile in case those are
        not given explicitly
    cuts_unit : str
        the unit of cut_left and cut_right, it can be seconds 's' or radians
        'rad'
    rf_station : Optional[RFStation]
        RFSectionParameters[0][0] is necessary for the conversion from radians
        to seconds if cuts_unit = 'rad'. RFSectionParameters[0][0] is the value
        of omega_rf of the main harmonic at turn number 0

    Attributes
    ----------
    cut_left : float
    cut_right : float
    n_slices : int
    n_sigma : float
    cuts_unit : str
    edges : float array
        contains the edges of the slices
    bin_centers : float array
        contains the centres of the slices

    Examples
    --------
    >>> from blond.input_parameters.ring import Ring
    >>> from blond.input_parameters.rf_parameters import RFStation
    >>> self.ring = Ring(n_turns = 1, ring_length = 100,
    >>> alpha = 0.00001, momentum = 1e9)
    >>> self.rf_params = RFStation(ring=self.ring, n_rf=1, harmonic=[4620],
    >>>                  voltage=[7e6], phi_rf_d=[0.])
    >>> CutOptions = profileModule.CutOptions(cut_left=0, cut_right=2*np.pi,
    >>> n_slices = 100, cuts_unit='rad', rf_station=self.rf_params)

    """

    @handle_legacy_kwargs
    def __init__(
        self,
        cut_left: Optional[float] = None,
        cut_right: Optional[float] = None,
        n_slices: int = 100,
        n_sigma: Optional[int] = None,
        cuts_unit: CutUnitType = "s",
        rf_station: Optional[RFStation] = None,
    ):
        """
        Constructor
        """

        self.cut_left: Optional[float] = (
            float(cut_left) if cut_left is not None else None
        )

        self.cut_right: Optional[float] = (
            float(cut_right) if cut_right is not None else None
        )

        self.n_slices = int(n_slices)

        self.n_sigma: Optional[float] = (
            float(n_sigma) if n_sigma is not None else None
        )

        self.cuts_unit: CutUnitType = cuts_unit

        self.rf_station: Optional[RFStation] = rf_station

        if self.cuts_unit == "rad" and self.rf_station is None:
            # CutError
            raise RuntimeError(
                'Argument "rf_station" required '
                + "convert from radians to seconds"
            )
        if self.cuts_unit not in ["rad", "s"]:
            # CutError
            raise NameError(
                f'cuts_unit should be "s" or "rad", not {cuts_unit=} !'
            )

        self.edges: NumpyArray = np.zeros(
            n_slices + 1, dtype=bm.precision.real_t, order="C"
        )
        self.bin_centers: NumpyArray = np.zeros(
            n_slices, dtype=bm.precision.real_t, order="C"
        )
        self.bin_size: float = 0.0
        # For CuPy backend
        self._device: DeviceType = "CPU"

    @property
    def RFParams(self):
        from warnings import warn

        warn(
            "AMBIGUOUS is deprecated, use ring",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rf_station

    @RFParams.setter
    def RFParams(self, val):
        from warnings import warn

        warn(
            "AMBIGUOUS is deprecated, use ring",
            DeprecationWarning,
            stacklevel=2,
        )
        self.rf_station = val

    def set_cuts(self, beam: Optional[Beam] = None):
        r"""
        Method to set self.cut_left, self.cut_right, self.edges and
        self.bin_centers attributes.
        The frame is defined by :math:`n\sigma_{RMS}` or manually by the user.
        If not, a default frame consisting of taking the whole bunch +5% of the
        maximum distance between two particles in the bunch will be taken
        in each side of the frame.
        """

        if self.cut_left is None and self.cut_right is None:
            if self.n_sigma is None:
                dt_min = beam.dt.min()
                dt_max = beam.dt.max()
                self.cut_left = dt_min - 0.05 * (dt_max - dt_min)
                self.cut_right = dt_max + 0.05 * (dt_max - dt_min)
            else:
                mean_coords = np.mean(beam.dt)
                sigma_coords = np.std(beam.dt)
                self.cut_left = mean_coords - self.n_sigma * sigma_coords / 2
                self.cut_right = mean_coords + self.n_sigma * sigma_coords / 2

        # todo handle cutleftNone, cutright!=None and vice versa

        else:
            self.cut_left = float(
                self.convert_coordinates(self.cut_left, self.cuts_unit)
            )
            self.cut_right = float(
                self.convert_coordinates(self.cut_right, self.cuts_unit)
            )

        self.edges = np.linspace(
            self.cut_left, self.cut_right, self.n_slices + 1
        ).astype(dtype=bm.precision.real_t, order="C", copy=False)
        self.bin_centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.bin_size = (self.cut_right - self.cut_left) / self.n_slices

    def track_cuts(self, beam: Beam):
        """
        Track the slice frame (limits and slice position) as the mean of the
        bunch moves.
        Requires Beam statistics!
        Method to be refined!
        """
        # todo Method to be refined!
        delta = beam.mean_dt - 0.5 * (self.cut_left + self.cut_right)

        self.cut_left += delta
        self.cut_right += delta
        self.edges += delta
        self.bin_centers += delta

    def convert_coordinates(
        self, value: float, input_unit_type: CutUnitType
    ) -> float:
        """
        Method to convert a value from 'rad' to 's'.
        """

        if input_unit_type == "s":
            return value

        elif input_unit_type == "rad":
            return value / float(
                self.rf_station.omega_rf[0, self.rf_station.counter[0]]
            )

        else:
            raise NameError(input_unit_type)

    def get_slices_parameters(
        self,
    ) -> tuple[int, float, float, None, NumpyArray, NumpyArray, float]:
        """
        Return all the computed parameters.
        """
        return (
            self.n_slices,
            self.cut_left,
            self.cut_right,
            self.n_sigma,
            self.edges,
            self.bin_centers,
            self.bin_size,
        )

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if self._device == "GPU":
            return

        # transfer recursively objects
        if recursive and self.rf_station:
            self.rf_station.to_gpu()

        import cupy as cp

        self.edges = cp.array(self.edges)
        self.bin_centers = cp.array(self.bin_centers)

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            return

        # transfer recursively objects
        if recursive and self.rf_station:
            self.rf_station.to_cpu()

        import cupy as cp

        self.edges = cp.asnumpy(self.edges)
        self.bin_centers = cp.asnumpy(self.bin_centers)

        if hasattr(self, "rf_voltage"):
            self.rf_voltage = cp.asnumpy(self.rf_voltage)

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"


class FitOptions:
    """
    This class defines the method to be used turn after turn to obtain the
    position and length of the bunch profile.

    Parameters
    ----------

    fit_option : string
        Current options are 'gaussian',
        'fwhm' (full-width-half-maximum converted to 4 sigma gaussian bunch)
        and 'rms'. The methods 'gaussian' and 'rms' give both 4 sigma.
    fit_extra_options : unknown # TODO
        For the moment no options can be passed into fitExtraOptions

    Attributes
    ----------

    fit_option : string
    fit_extra_options : unknown # TODO
    """

    @handle_legacy_kwargs
    def __init__(
        self,
        fit_option: Optional[FitOptionTypes] = None,
        fit_extra_options: None = None,
    ):  # todo type hint
        """
        Constructor
        """

        self.fit_option: Optional[FitOptionTypes] = fit_option
        self.fit_extra_options: None = fit_extra_options

    @property
    def fitExtraOptions(self):
        from warnings import warn

        warn(
            "fitExtraOptions is deprecated, use fit_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fit_extra_options

    @fitExtraOptions.setter
    def fitExtraOptions(self, val):
        from warnings import warn

        warn(
            "fitExtraOptions is deprecated, use fit_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        self.fit_extra_options = val


class FilterOptions:
    """
    This class defines the filter to be used turn after turn to smooth
    the bunch profile.

    Parameters
    ----------

    filter_method : string
        The only option available is 'chebishev'
    filter_extra_options : dictionary
        Parameters for the Chebishev filter (see the method
        beam_profile_filter_chebyshev in filters_and_fitting.py in the toolbox
        package)

    Attributes
    ----------

    filter_method : string
    filter_extra_options : dictionary

    """

    @handle_legacy_kwargs
    def __init__(
        self,
        filter_method: Optional[FilterMethodType] = None,
        filter_extra_options: Optional[FilterExtraOptionsType] = None,
    ):
        """
        Constructor
        """

        self.filter_method: Optional[FilterMethodType] = filter_method
        self.filter_extra_options: Optional[FilterExtraOptionsType] = (
            filter_extra_options
        )

    @property
    def filterMethod(self):
        from warnings import warn

        warn(
            "filterMethod is deprecated, use filter_method",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter_method

    @filterMethod.setter
    def filterMethod(self, val):
        from warnings import warn

        warn(
            "filterMethod is deprecated, use filter_method",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filter_method = val

    @property
    def filterExtraOptions(self):
        from warnings import warn

        warn(
            "filterExtraOptions is deprecated, use filter_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter_extra_options

    @filterExtraOptions.setter
    def filterExtraOptions(self, val):
        from warnings import warn

        warn(
            "filterExtraOptions is deprecated, use filter_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filter_extra_options = val


class OtherSlicesOptions:
    """
    This class groups all the remaining options for the Profile class.

    Parameters
    ----------

    smooth : boolean
        If set True, this method slices the bunch not in the
        standard way (fixed one slice all the macroparticles contribute
        with +1 or 0 depending on if they are inside or not). The method assigns
        to each macroparticle a real value between 0 and +1 depending on its
        time coordinate. This method can be considered a filter able to smooth
        the profile.
    direct_slicing : boolean
        If set True, the profile is calculated when the Profile class below
        is created. If False the user has to manually track the Profile object
        in the main file after its creation

    Attributes
    ----------

    smooth : boolean
    direct_slicing : boolean

    """

    def __init__(self, smooth: bool = False, direct_slicing: bool = True):
        """
        Constructor
        """

        self.smooth = smooth
        self.direct_slicing = direct_slicing


class Profile:
    """
    Contains the beam profile and related quantities including beam spectrum,
    profile derivative.

    Parameters
    ----------

    beam : Beam
        Beam from which the profile has to be calculated
    cut_options : CutOptions, Optional
        Options for profile cutting (see above)
    fit_options : FitOptions, Optional
        Options to get profile position and length (see above)
    filter_options : FilterOptions, Optional
        Options to set a filter (see above)
    other_slices_options : OtherSlicesOptions, Optional
        All remaining options, like smooth histogram and direct
        slicing (see above)

    Attributes
    ----------

    beam: Beam
    n_slices : int
        number of slices to be used
    cut_left : float
        left extreme of the profile
    cut_right : float
        right extreme of the profile
    n_sigma : float
        defines the left and right extremes of the profile in case those are
        not given explicitly
    edges : float array
        contains the edges of the slices
    bin_centers : NumpyArray, optional
        contains the centres of the slices
    bin_size : float
        lenght of one bin (or slice)
    n_macroparticles : float array
        contains the histogram (or profile); its elements are real if the
        smooth histogram tracking is used
    beam_spectrum : float array
        contains the spectrum of the beam (arb. units)
    beam_spectrum_freq : float array
        contains the frequencies on which the spectrum is computed [Hz]
    operations : list
        contains all the methods to be called every turn, like slice track,
        fitting, filtering etc.
    bunchPosition : float
        profile position [s]
    bunchLength : float
        profile length [s]
    filter_extra_options : unknown (see above)

    Examples
    --------

    >>> n_slices = 100
    >>> CutOptions = profileModule.CutOptions(cut_left=0,
    >>>       cut_right=self.ring.t_rev[0], n_slices = n_slices, cuts_unit='s')
    >>> FitOptions = profileModule.FitOptions(fit_option='gaussian',
    >>>                                        fit_extra_options=None)
    >>> filter_option = {'pass_frequency':1e7,
    >>>    'stop_frequency':1e8, 'gain_pass':1, 'gain_stop':2,
    >>>    'transfer_function_plot':False}
    >>> FilterOptions = profileModule.FilterOptions(filter_method='chebishev',
    >>>         filter_extra_options=filter_option)
    >>> OtherSlicesOptions = profileModule.OtherSlicesOptions(smooth=False,
    >>>                             direct_slicing = True)
    >>> self.profile4 = profileModule.Profile(my_beam, cut_options = CutOptions,
    >>>                     fit_options= FitOptions,
    >>>                     filter_options=FilterOptions,
    >>>                     other_slices_options = OtherSlicesOptions)

    """

    @handle_legacy_kwargs
    def __init__(
        self,
        beam: Beam,
        cut_options: Optional[CutOptions] = None,
        fit_options: Optional[FitOptions] = None,
        filter_options: Optional[FilterOptions] = None,
        other_slices_options: Optional[OtherSlicesOptions] = None,
    ):
        """
        Constructor
        """

        if cut_options is None:
            cut_options = CutOptions()
        if fit_options is None:
            fit_options = FitOptions()
        if filter_options is None:
            filter_options = FilterOptions()
        if other_slices_options is None:
            other_slices_options = OtherSlicesOptions()

        # Copy of CutOptions object to be used for re-slicing
        self.cut_options = cut_options

        # Define bins
        cut_options.set_cuts(beam)

        # Import (reference) Beam
        self.beam = beam

        self.n_slices = 0
        self.cut_left = 0.0
        self.cut_right = 0.0
        self.n_sigma = 0
        self.edges: NumpyArray | None = None
        self.bin_centers: NumpyArray | None = None
        self.bin_size = 0.0
        self.fit_extra_options = None  # todo typing
        # Get all computed parameters from CutOptions
        self.set_slices_parameters()

        # Initialize profile array as zero array
        self.n_macroparticles: NumpyArray = np.zeros(
            self.n_slices, dtype=bm.precision.real_t, order="C"
        )

        # Initialize beam_spectrum and beam_spectrum_freq as empty arrays
        self.beam_spectrum: NumpyArray = np.array(
            [], dtype=bm.precision.real_t, order="C"
        )
        self.beam_spectrum_freq: NumpyArray = np.array(
            [], dtype=bm.precision.real_t, order="C"
        )

        self.operations: list[Callable] = []
        if other_slices_options.smooth:
            self.operations.append(self._slice_smooth)
        else:
            self.operations.append(self._slice)

        self.fit_option = fit_options.fit_option
        if (
            fit_options.fit_option is not None
        ):  # todo remove conditional attributes
            self.bunchPosition = 0.0
            self.bunchLength = 0.0
            if fit_options.fit_option == "gaussian":
                self.operations.append(self.apply_fit)
            elif fit_options.fit_option == "rms":
                self.operations.append(self.rms)
            elif fit_options.fit_option == "fwhm":
                self.operations.append(self.fwhm)
            else:
                raise NameError(f"{fit_options=}")

        if filter_options.filter_method == "chebishev":
            self.filter_extra_options = filter_options.filter_extra_options
            self.operations.append(self.apply_filter)
        elif filter_options.filter_method is None:
            pass
        else:
            raise NameError(f"{filter_options.filter_method=}")

        if other_slices_options.direct_slicing and self.beam is not None:
            self.track()

        # For CuPy backend
        self._device: DeviceType = "CPU"

    @property
    def Beam(self):
        from warnings import warn

        warn("Beam is deprecated, use beam", DeprecationWarning, stacklevel=2)
        return self.beam

    @Beam.setter
    def Beam(self, val):
        from warnings import warn

        warn("Beam is deprecated, use beam", DeprecationWarning, stacklevel=2)
        self.beam = val

    @property
    def filterExtraOptions(self):
        from warnings import warn

        warn(
            "filterExtraOptions is deprecated, use filter_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.filter_extra_options

    @filterExtraOptions.setter
    def filterExtraOptions(self, val):
        from warnings import warn

        warn(
            "filterExtraOptions is deprecated, use filter_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filter_extra_options = val

    def set_slices_parameters(self):
        """
        Set various slices parameters.
        """
        (
            self.n_slices,
            self.cut_left,
            self.cut_right,
            self.n_sigma,
            self.edges,
            self.bin_centers,
            self.bin_size,
        ) = self.cut_options.get_slices_parameters()  # fixme get_slices_parameters doesnt exist

    def track(self):
        """
        Track method in order to update the slicing along with the tracker.
        The kwargs are currently only needed to forward the reduce kw argument
        needed for the MPI version.
        """

        for operation in self.operations:
            operation()

    def _slice(self):
        """
        Constant space slicing with a constant frame.
        """
        bm.slice_beam(
            self.beam.dt, self.n_macroparticles, self.cut_left, self.cut_right
        )

        if bm.in_mpi():
            self.reduce_histo()

    def reduce_histo(self, dtype=np.uint32):
        """
        Aggregate all local histograms to calculate the global beam profile.
        """
        if not bm.in_mpi():
            raise RuntimeError(
                "ERROR: Cannot use this routine unless in MPI Mode"
            )

        from ..utils.mpi_config import WORKER

        if WORKER.workers == 1:
            return

        if self.beam._mpi_is_splitted:
            if "CPU" in bm.device:
                # Convert to uint32t for better performance
                self.n_macroparticles = self.n_macroparticles.astype(
                    dtype, order="C"
                )

            if bm.device == "GPU":
                import cupy as cp

                # tranfer to cpu
                self.n_macroparticles = cp.asnumpy(
                    self.n_macroparticles, dtype=dtype
                )

            WORKER.allreduce(self.n_macroparticles)

            if bm.device == "GPU":
                # transfer back to gpu
                self.n_macroparticles = cp.array(
                    self.n_macroparticles, dtype=bm.precision.real_t
                )

            if "CPU" in bm.device:
                # Convert back to float64
                self.n_macroparticles = self.n_macroparticles.astype(
                    dtype=bm.precision.real_t, order="C", copy=False
                )

    def _slice_smooth(self, reduce: bool = True):
        """
        At the moment 4x slower than _slice but smoother (filtered).
        """
        bm.slice_smooth(
            self.beam.dt, self.n_macroparticles, self.cut_left, self.cut_right
        )

        if bm.in_mpi():
            self.reduce_histo(dtype=np.float64)

    def apply_fit(self):
        """
        It applies Gaussian fit to the profile.
        """

        if self.bunchLength == 0:
            p_0 = [
                float(self.n_macroparticles.max()),
                float(self.beam.dt.mean()),
                float(self.beam.dt.std()),
            ]
        else:
            p_0 = [
                float(self.n_macroparticles.max()),
                float(self.bunchPosition),
                float(self.bunchLength / 4.0),
            ]

        self.fit_extra_options = ffroutines.gaussian_fit(
            self.n_macroparticles, self.bin_centers, p_0
        )
        self.bunchPosition = self.fit_extra_options[1]
        self.bunchLength = 4 * self.fit_extra_options[2]

    @property
    def fitExtraOptions(self):  # TODO
        from warnings import warn

        warn(
            "fitExtraOptions is deprecated, use fit_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )  # TODO
        return self.fit_extra_options

    @fitExtraOptions.setter  # TODO
    def fitExtraOptions(self, val):  # TODO
        from warnings import warn

        warn(
            "fitExtraOptions is deprecated, use fit_extra_options",
            DeprecationWarning,
            stacklevel=2,
        )  # TODO
        self.fit_extra_options = val

    def apply_filter(self):
        """
        It applies Chebishev filter to the profile.
        """
        self.n_macroparticles = ffroutines.beam_profile_filter_chebyshev(
            self.n_macroparticles, self.bin_centers, self.filter_extra_options
        )

    def rms(self):
        """
        Computation of the RMS bunch length and position from the line
        density (bunch length = 4sigma).
        """

        self.bunchPosition, self.bunchLength = ffroutines.rms(
            self.n_macroparticles, self.bin_centers
        )

    def rms_multibunch(
        self,
        n_bunches,
        bunch_spacing_buckets,
        bucket_size_tau,
        bucket_tolerance=0.40,
    ):
        """
        Computation of the bunch length (4sigma) and position from RMS.
        """

        self.bunchPosition, self.bunchLength = ffroutines.rms_multibunch(
            self.n_macroparticles,
            self.bin_centers,
            n_bunches,
            bunch_spacing_buckets,
            bucket_size_tau,
            bucket_tolerance,
        )

    def fwhm(self, shift: float = 0):
        """
        Computation of the bunch length and position from the FWHM
        assuming Gaussian line density.
        """

        self.bunchPosition, self.bunchLength = ffroutines.fwhm(
            self.n_macroparticles, self.bin_centers, shift
        )

    def fwhm_multibunch(
        self,
        n_bunches,
        bunch_spacing_buckets,
        bucket_size_tau,
        bucket_tolerance=0.40,
        shift=0,
    ):
        """
        Computation of the bunch length and position from the FWHM
        assuming Gaussian line density for multibunch case.
        """

        self.bunchPosition, self.bunchLength = ffroutines.fwhm_multibunch(
            self.n_macroparticles,
            self.bin_centers,
            n_bunches,
            bunch_spacing_buckets,
            bucket_size_tau,
            bucket_tolerance,
            shift,
        )

    def beam_spectrum_freq_generation(self, n_sampling_fft: int):
        """
        Frequency array of the beam spectrum
        """

        self.beam_spectrum_freq = bm.rfftfreq(n_sampling_fft, self.bin_size)

    def beam_spectrum_generation(self, n_sampling_fft: int):
        """
        Beam spectrum calculation
        """
        self.beam_spectrum = bm.rfft(self.n_macroparticles, n_sampling_fft)

    def beam_profile_derivative(
        self, mode: BeamProfileDerivativeModes = "gradient"
    ) -> Tuple[NumpyArray, NumpyArray]:
        """
        The input is one of the three available methods for differentiating
        a function. The two outputs are the bin centres and the discrete
        derivative of the Beam profile respectively.*
        """

        bin_centers = self.bin_centers
        dist_centers = bin_centers[1] - bin_centers[0]

        if mode == "filter1d":
            if bm.device == "GPU":
                raise RuntimeError("filter1d mode is not supported in GPU.")

            derivative = (
                ndimage.gaussian_filter1d(
                    self.n_macroparticles, sigma=1, order=1, mode="wrap"
                )
                / dist_centers
            )
        elif mode == "gradient":
            derivative = bm.gradient(self.n_macroparticles, dist_centers)
        elif mode == "diff":
            derivative = bm.diff(self.n_macroparticles) / dist_centers
            diffCenters = bin_centers[0:-1] + dist_centers / 2
            derivative = bm.interp(bin_centers, diffCenters, derivative)
        else:
            # ProfileDerivativeError
            raise RuntimeError("Option for derivative is not recognized.")

        return bin_centers, derivative

    def to_gpu(self, recursive: bool = True):
        """
        Transfer all necessary arrays to the GPU
        """
        # Check if to_gpu has been invoked already
        if hasattr(self, "_device") and self._device == "GPU":
            return

        # transfer recursively objects to_gpu
        if recursive and self.beam:
            self.beam.to_gpu()

        if recursive and self.cut_options:
            self.cut_options.to_gpu()

        import cupy as cp

        self.bin_centers = self.cut_options.bin_centers
        self.edges = self.cut_options.edges

        self.n_macroparticles = cp.array(self.n_macroparticles)
        self.beam_spectrum = cp.array(self.beam_spectrum)
        self.beam_spectrum_freq = cp.array(self.beam_spectrum_freq)

        # to make sure it will not be called again
        self._device: DeviceType = "GPU"

    def to_cpu(self, recursive=True):
        """
        Transfer all necessary arrays back to the CPU
        """
        # Check if to_cpu has been invoked already
        if hasattr(self, "_device") and self._device == "CPU":
            return

        # transfer recursively objects
        if recursive and self.beam:
            self.beam.to_cpu()

        if recursive and self.cut_options:
            self.cut_options.to_cpu()

        import cupy as cp

        self.bin_centers = self.cut_options.bin_centers
        self.edges = self.cut_options.edges

        self.n_macroparticles = cp.asnumpy(self.n_macroparticles)
        self.beam_spectrum = cp.asnumpy(self.beam_spectrum)
        self.beam_spectrum_freq = cp.asnumpy(self.beam_spectrum_freq)

        # to make sure it will not be called again
        self._device: DeviceType = "CPU"
