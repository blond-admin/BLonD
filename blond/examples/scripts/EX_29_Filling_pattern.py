# Copyright CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENSE.txt.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

# pragma: no cover
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from blond import Batch, FillingPattern, Train
from blond.cycles.filling_patterns import plot
from blond.specifics.cern.lhc import filling_pattern_from_scheme_file


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
    print(f"  batch[:8]:      {pattern.label('batch')[:8]}")
    print(f"  train[:8]:      {pattern.label('train')[:8]}")
    print(f"  injection[:8]:  {pattern.label('injection')[:8]}")

    # Per-bunch quantity arrays support numpy-masked assignment.
    pattern.intensity = np.ones(pattern.n_bunches) * 1.1e11
    pattern.intensity[pattern.label("batch") == 0] = 0.5e11
    pattern.intensity[pattern.label("injection") == 1] = 0.8e11
    print(f"  intensity[:8]:  {pattern.intensity[:8]}")

    plot(pattern)

    # Real machine fills: import an official LHC filling-scheme file
    # directly (scheme names are lossy labels — always load the JSON).
    scheme_path = (
        Path(__file__).parent
        / "resources"
        / "EX_29"
        / "25ns_1500b_1488_684_729_240bpi_8inj_HItests_Fill4.json"
    )
    lhc_pattern = filling_pattern_from_scheme_file(path=scheme_path, beam=1)
    print(lhc_pattern)
    print(
        f"  n_injections={lhc_pattern.n_groups('injection')}, "
        f"n_batches={lhc_pattern.n_groups('batch')}"
    )
    plot(lhc_pattern, face_label="injection", edge_label="batch")


if __name__ == "__main__":  # pragma: no cover
    main()
    plt.show()
