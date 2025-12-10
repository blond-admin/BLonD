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
    "ConstantMagneticCycle": CYCLE,
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


def main():
    """Reads all imports that are available at BLonD toplevel and creates RST file."""
    blond_toplevel_variable_names = dir(
        blond
    )  # All variable names in `blond.__init__.py`

    # Prepare dict: category → list of RST blocks
    categorized_entries = {cat: [] for cat in CATEGORIES}
    print_unlinked_classes(blond_toplevel_variable_names)
    for name in blond_toplevel_variable_names:
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

            category = ASSIGNED_CATEGORIES[name]
            categorized_entries[category].append(block)

    rst_content = make_rst_string(categorized_entries)

    with open("modules/blond_main_objects.rst", "w", encoding="utf-8") as f:
        f.write(rst_content)
    print('Created "modules/blond_main_objects.rst"')


def make_rst_string(categorized_entries: dict[str, list[str]]):
    """Create a string for in RST style from the input blocks.

    Parameters
    ----------
    categorized_entries
        Each category has a list of strings to be added per category.
    """
    rst_file = """
BLonD Main Objects
==================

.. toctree::
   :maxdepth: 1
"""
    for category, blocks in categorized_entries.items():
        if len(blocks) == 0:
            continue  # skip empty categories

        rst_file += f"""

{category}
{"-" * len(category)}
"""
        rst_file += "\n".join(blocks)
    return rst_file


def print_unlinked_classes(blond_toplevel_variable_names: list[str]):
    """Print all objects that are not listed in `ASSIGNED_CATEGORIES`.

    Parameters
    ----------
    blond_toplevel_variable_names
        All variable names in `blond.__init__.py`
    """
    assigned_category_keys = ASSIGNED_CATEGORIES.keys()
    for name in blond_toplevel_variable_names:
        obj = getattr(blond, name)
        if inspect.isclass(obj) and name not in assigned_category_keys:
            print(
                f"{name} is missing in `ASSIGNED_CATEGORIES`."
                f" Please add it in `{__file__}`."
            )


if __name__ == "__main__":
    main()
