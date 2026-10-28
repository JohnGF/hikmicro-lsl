"""Stream HIKMICRO Pocket 2 / XTherm-family thermal data into LSL.

Public surface is re-exported lazily from submodules so that importing the
package never pulls in pylsl, OpenCV or the camera device.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .radiometry import (  # noqa: F401
    FRAME_HEIGHT,
    FRAME_U16,
    HEIGHT,
    META_ROWS,
    PIXELS,
    TABLE_SIZE,
    WIDTH,
    Meta,
    frame_to_celsius,
    iron_palette,
    parse_meta,
    temp_table,
)

__all__ = [
    "FRAME_HEIGHT",
    "FRAME_U16",
    "HEIGHT",
    "META_ROWS",
    "PIXELS",
    "ROI_CHANNELS",
    "ROI_LABELS",
    "TABLE_SIZE",
    "WIDTH",
    "Meta",
    "frame_to_celsius",
    "iron_palette",
    "parse_meta",
    "roi_vector",
    "temp_table",
    "__version__",
]


def __getattr__(name: str):
    # Lazily re-export the streamer surface so importing the package never
    # pulls in pylsl.
    if name in {"ROI_CHANNELS", "ROI_LABELS", "roi_vector"}:
        from . import lsl_streamer

        return getattr(lsl_streamer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
