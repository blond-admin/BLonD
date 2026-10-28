"""
Render the BLonD logo SVG to the PNG variants used in the docs/README.

Usage::

    pip install resvg-py
    python resources/convert_logo.py

Writes, next to this script:

- ``BLonD3_centered.png``: 696x696, transparent background
- ``BLonD3_centered_white_background_192x192.png``: 192x192, white
"""

from pathlib import Path

import resvg_py

HERE = Path(__file__).parent
SVG = HERE / "BLonD3_logo.svg"

VARIANTS = (
    ("BLonD3_centered.png", 696, None),
    ("BLonD3_centered_white_background_192x192.png", 192, "#ffffff"),
)


def main():
    """Render every PNG variant from the logo SVG."""
    for name, size, background in VARIANTS:
        png = resvg_py.svg_to_bytes(
            svg_path=str(SVG),
            width=size,
            height=size,
            background=background,
        )
        (HERE / name).write_bytes(bytes(png))
        print(f"wrote {name} ({size}x{size})")


if __name__ == "__main__":
    main()
