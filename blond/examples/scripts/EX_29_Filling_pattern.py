# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover
import numpy as np
from matplotlib import pyplot as plt

from blond.cycles.filling_patterns import Batch, FillingPattern, Train, plot


def main():
    # LHC-like nesting: PS batch -> SPS train -> LHC injection -> full ring
    ps_batch = Batch(n_bunches=72, bunch_gap=9)  # 25 ns bunch spacing
    sps_train = Train(unit=ps_batch, n_copies=4, copy_gap=8)
    injection = sps_train.with_label("injection")
    pattern = FillingPattern(
        injection.with_trailing_gap(38) * 11 + injection,
        harmonic_number=35640,
    )

    print(pattern)
    print(
        f"  n_batches={pattern.n_groups('batch')}, "
        f"n_trains={pattern.n_groups('train')}, "
        f"n_injections={pattern.n_groups('injection')}"
    )
    print(f"  bucket_indices[:8]:  {pattern.bucket_indices[:8]}")
    print(f"  batch[:8]:      {pattern.tier('batch')[:8]}")
    print(f"  train[:8]:      {pattern.tier('train')[:8]}")
    print(f"  injection[:8]:  {pattern.tier('injection')[:8]}")

    # Per-bunch property arrays support numpy-masked assignment.
    pattern.intensity = np.ones(pattern.n_bunches) * 1.1e11
    pattern.intensity[pattern.tier("batch") == 0] = 0.5e11
    pattern.intensity[pattern.tier("injection") == 1] = 0.8e11
    print(f"  intensity[:8]:  {pattern.intensity[:8]}")

    plot(pattern)


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
