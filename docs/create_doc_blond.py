"""Adds the API Documentation to `blond.rst`."""

import os

content = """
API Documentation
=================

.. toctree::
   blond.core
   blond.generals
   blond.acc_math
   blond.beam_preparation
   blond.convenience
   blond.cycles
   blond.examples
   blond.experimental
   blond.handle_results
   blond.interfaces
   blond.legacy
   blond.performance_blond3
   blond.physics
   blond.specifics
"""


def main():
    """Create blond.rst."""
    this_folder = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(this_folder, "modules/blond.rst")
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
    print(f'Created "{target}".')


if __name__ == "__main__":
    main()
