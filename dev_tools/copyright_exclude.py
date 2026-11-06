"""Filenames of vendored files that must keep their upstream header.

These are third-party files that are not ours; we must not add our
license. e.g. the C++ sources hardcopied from CERN's external
rf-noise-cpp project.
"""

THIRD_PARTY_FILENAMES = frozenset(
    (
        "rf_noise_wrapper.cpp",
        "varigen.h",
    )
)
