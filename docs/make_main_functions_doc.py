"""Creates `blond_main_objects.rst` as sorted documentation.

Notes
-----
This script is intended to crash if a item appears without an
`ASSIGNED_CATEGORIES`.
"""

import inspect

import blond

# Category Groups definition (can be extended if needed)
LATTICE = "Lattice & Hardware"
CYCLE = "Energy Ramp"
BEAM = "Beam Generation & Distribution"
DYNAMICS = "Beam Dynamics & Tracking"
DIAGNOSTICS = "Observables & Diagnostics"
PLOTTING = "Plotting & Visualization"
BACKEND = "Computing Backend"
MISC = "Misc"

# define display order
CATEGORIES = (
    LATTICE,
    CYCLE,
    BEAM,
    DYNAMICS,
    DIAGNOSTICS,
    PLOTTING,
    BACKEND,
    MISC,
)

ASSIGNED_CATEGORIES = {
    # Lattice & Hardware
    "Simulation": LATTICE,
    "BoxLosses": LATTICE,
    "ConstantMagneticCycle": LATTICE,
    "DriftSimple": LATTICE,
    "MultiHarmonicRfStation": LATTICE,
    "ReferenceEnergyChange": LATTICE,
    "Ring": LATTICE,
    "SingleHarmonicRfStation": LATTICE,
    "UserDefinedElement": LATTICE,
    "WakeField": LATTICE,
    "StaticProfile": LATTICE,
    # Cycle
    "MagneticCycleByTime": CYCLE,
    "MagneticCyclePerTurn": CYCLE,
    "MagneticCyclePerTurnAllRfStations": CYCLE,
    # Beam Generation & Distribution
    "Beam": BEAM,
    "BiGaussian": BEAM,
    # Diagnostics
    "RfStationPhaseObservation": DIAGNOSTICS,
    "StaticProfileObservation": DIAGNOSTICS,
    "BeamObservationInRingElement": DIAGNOSTICS,
    "BeamObservationOncePerTurn": DIAGNOSTICS,
    # Plotting
    "AllowPlotting": PLOTTING,
    # Backend / Precision
    "Cupy32Bit": BACKEND,
    "Cupy64Bit": BACKEND,
    "Numpy32Bit": BACKEND,
    "Numpy64Bit": BACKEND,
}


def assign_category(class_name):
    """Return the category name for a given class, or OTHER_CATEGORY_NAME."""
    return ASSIGNED_CATEGORIES[
        class_name
    ]  # intended to fail whn category is undefined


def main():
    """Reads all imports that are available at BLonD toplevel and creates RST file."""
    alls = dir(blond)

    # Prepare dict: category → list of RST blocks
    categorized_entries = {cat: [] for cat in CATEGORIES}
    for name in alls:
        obj = getattr(blond, name)
        if inspect.isclass(obj):
            try:
                assign_category(name)
            except KeyError as exc:
                print(str(exc))
    for name in alls:
        obj = getattr(blond, name)

        if inspect.isclass(obj):
            module = obj.__module__
            qualname = obj.__qualname__
            full_path = f"{module}.{qualname}"

            block = f"""
{name}
{"~" * len(name)}

.. autoclass:: {full_path}
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:

"""

            category = assign_category(name)
            categorized_entries[category].append(block)

    # ------------------------------------------------------------
    # Build RST file
    # ------------------------------------------------------------
    rst_file = """
BLonD Main Objects
==================

.. toctree::
   :maxdepth: 1
"""

    for category, blocks in categorized_entries.items():
        if not blocks:
            continue  # skip empty categories

        rst_file += f"""

{category}
{"-" * len(category)}
"""
        rst_file += "\n".join(blocks)

    with open("modules/blond_main_objects.rst", "w", encoding="utf-8") as f:
        f.write(rst_file)
    print('Created "modules/blond_main_objects.rst"')


if __name__ == "__main__":
    main()
