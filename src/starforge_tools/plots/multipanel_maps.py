"""
Contains the implementation of different panel rendering maps.
"""

from meshoid import Meshoid
import numpy as np


def SurfaceDensity(pdata: dict, meshoid: Meshoid, mapargs: dict):
    return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)


def VelocityDispersion(pdata: dict, meshoid: Meshoid, mapargs: dict):
    sigma = meshoid.SurfaceDensity(pdata, **mapargs)
    mvSqr_map = (
        meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2] ** 2, **mapargs) / sigma
    )
    mvmap = meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2], **mapargs) / sigma
    return np.clip(mvSqr_map - mvmap**2, 0, 1e100)


def MassWeightedTemperature(pdata: dict, meshoid: Meshoid, mapargs: dict):
    return meshoid.ProjectedAverage(pdata["PartType0/Masses"], **mapargs)


def AlfvenSpeed(pdata: dict, meshoid: Meshoid, mapargs: dict):
    return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
