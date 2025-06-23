from typing import TYPE_CHECKING

import numpy as np

from ..backend import Specials

if TYPE_CHECKING:  # pragma: no cover

    from numpy.typing import NDArray as NumpyArray
    from cupy.typing import NDArray as CupyArray

class CudaSpecials(Specials):
    pass