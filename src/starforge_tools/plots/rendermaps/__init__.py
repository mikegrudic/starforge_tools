"""Submodule in which gas render maps are defined"""

from .SurfaceDensity import SurfaceDensity
from .AlfvenSpeed import AlfvenSpeed
from .MagneticEnergyFraction import MagneticEnergyFraction
from .MassWeightedTemperature import MassWeightedTemperature
from .VelocityDispersion import VelocityDispersion
from .XCoordinate import XCoordinate
from .ZCoordinate import ZCoordinate


DEFAULT_MAPS = (
    "SurfaceDensity",
    "VelocityDispersion",
    "MassWeightedTemperature",
    "MagneticEnergyFraction",
)
