from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from os import PathLike

import numpy as np
from numpy.typing import NDArray as NumpyArray


class ImpedanceReader(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod  # pragma: no cover
    def load_file(self, filepath: PathLike) -> tuple[NumpyArray, NumpyArray]:
        return freq, amplitude  # NOQA


class CsvReader(ImpedanceReader):
    def __init__(self, **kwargs) -> None:
        """Simple CSV file reader for two rows of data.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments for `numpy.loadtxt`
        """
        super().__init__()
        self.kwargs = kwargs

    def load_file(self, filepath: PathLike) -> tuple[NumpyArray, NumpyArray]:
        data = np.loadtxt(filepath, **self.kwargs)
        return data[:, 0], data[:, 1]


class ExampleImpedanceReader1(ImpedanceReader):
    def __init__(self):
        super().__init__()

    def load_file(self, filepath: PathLike) -> tuple[NumpyArray, NumpyArray]:
        table = np.loadtxt(
            filepath,
            skiprows=1,
            dtype=complex,
            encoding="utf-8",
            converters={
                0: lambda s: complex(
                    bytes(s, encoding="utf-8")
                    .decode("UTF-8")
                    .replace("i", "j")
                ),
                1: lambda y: complex(
                    bytes(y, encoding="utf-8")
                    .decode("UTF-8")
                    .replace("i", "j")
                ),
            },
        )
        freq, amplitude = table[:, 0].real, table[:, 1]
        return freq, amplitude


class ModesExampleReader2(str, Enum):
    OPEN_LOOP = "open loop"
    CLOSED_LOOP = "closed loop"
    SHORTED = "shorted"


class ExampleImpedanceReader2(ImpedanceReader):
    def __init__(
        self, mode: ModesExampleReader2 = ModesExampleReader2.CLOSED_LOOP
    ):
        super().__init__()
        self._mode = mode

    def load_file(self, filepath: PathLike) -> tuple[NumpyArray, NumpyArray]:
        data = np.loadtxt(filepath, dtype=float, skiprows=1)
        data[:, 3] = np.deg2rad(data[:, 3])
        data[:, 5] = np.deg2rad(data[:, 5])
        data[:, 7] = np.deg2rad(data[:, 7])

        freq_x = data[:, 0]
        if self._mode.value == ModesExampleReader2.OPEN_LOOP.value:
            Re_Z = data[:, 4] * np.cos(data[:, 3])
            Im_Z = data[:, 4] * np.sin(data[:, 3])
        elif self._mode.value == ModesExampleReader2.CLOSED_LOOP.value:
            Re_Z = data[:, 2] * np.cos(data[:, 5])
            Im_Z = data[:, 2] * np.sin(data[:, 5])
        elif self._mode.value == ModesExampleReader2.SHORTED.value:
            Re_Z = data[:, 6] * np.cos(data[:, 7])
            Im_Z = data[:, 6] * np.sin(data[:, 7])
        else:
            raise NameError(f"{self._mode=}")
        scale = 13
        freq_y = scale * (Re_Z + 1j * Im_Z)

        return freq_x, freq_y
