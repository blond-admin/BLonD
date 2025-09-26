from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    if sys.version_info < (3, 11):  # todo consider this fix in other files
        from typing_extensions import TypedDict, NotRequired
    else:
        from typing import TypedDict, NotRequired

    from typing import TYPE_CHECKING, Literal, Protocol

    from numpy.typing import NDArray as NumpyArray

    class DistributionOptionsType(TypedDict):
        type: NotRequired[str]
        exponent: NotRequired[float]
        emittance: NotRequired[float]
        bunch_length: NotRequired[float]
        bunch_length_fit: NotRequired[BunchLengthFitTypes]
        density_variable: NotRequired[DistributionVariableType]

    class DistributionUserTableType(TypedDict):
        user_table_action: NumpyArray
        user_table_distribution: NumpyArray

    class LineDensityInputType(TypedDict):
        time_line_den: NumpyArray
        line_density: NumpyArray

    class ExtraVoltageDictType(TypedDict):
        time_array: NumpyArray
        voltage_array: NumpyArray

    class FilterExtraOptionsType(TypedDict):
        pass_frequency: float
        stop_frequency: float
        gain_pass: float
        gain_stop: float
        transfer_function_plot: NotRequired[bool]

    DistributionVariableType = Literal["Action", "Hamiltonian"]

    HalfOptionType = Literal["first", "second", "both"]

    BunchLengthFitTypes = Literal["full", "gauss", "fwhm"] | None

    LineDensityDistType = Literal[
        "waterbag",
        "parabolic_amplitude",
        "parabolic_line",
        "binomial",
        "gaussian",
        "cosine_squared",
    ]

    DistributionFunctionDistType = Literal[
        "waterbag",
        "parabolic_amplitude",
        "parabolic_line",
        "binomial",
        "gaussian",
    ]

    DeviceType = Literal["CPU", "GPU"]

    CutUnitType = Literal["rad", "s"]

    FitOptionTypes = Literal["gaussian", "fwhm", "rms"]

    FilterMethodType = Literal["chebishev"]

    ResonatorsMethodType = Literal["c++", "python"]
    InterpolationTypes = Literal["linear", "cubic", "derivative", "akima"]

    BeamProfileDerivativeModes = Literal[
        "filter1d",
        "gradient",
        "diff",
    ]

    class Trackable(Protocol):
        def track(self): ...


DistTypeDistFunction = Literal[
    "waterbag", "parabolic_amplitude", "parabolic_line", "binomial", "gaussian"
]
