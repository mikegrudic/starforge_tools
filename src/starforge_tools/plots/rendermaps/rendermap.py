"""Rendermap class definition: provides a method for rendering the map"""

from meshoid import Meshoid
import numpy as np

MINIMAL_FIELDS = (
    "PartType0/Coordinates",
    "PartType0/SmoothingLength",
)


class RenderMap:
    plotlabel = ""
    colormap = ""
    required_datafields = MINIMAL_FIELDS
    cmap_default_limits = (None, None)
    log_cache = True  # store log10(values) in float16 in the disk cache; override to False for signed quantities

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        """Trivial default render method"""
        return np.zeros((mapargs["res"], mapargs["res"]))
