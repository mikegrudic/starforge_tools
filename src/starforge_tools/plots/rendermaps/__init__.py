"""Submodule in which gas render maps are defined"""

from . import (
    SurfaceDensity,
    AlfvenSpeed,
    MassWeightedTemperature,
    VelocityDispersion,
    XCoordinate,
    ZCoordinate,
    #    RenderMap,
)


DEFAULT_MAPS = (
    "SurfaceDensity",
    "VelocityDispersion",
    "MassWeightedTemperature",
    "AlfvenSpeed",
    #    "XCoordinate",
    #  "ZCoordinate",
)
