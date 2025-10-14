"""
Contains the implementation of different panel rendering maps.
"""

from meshoid import Meshoid
import numpy as np


class RenderMap:
    def __init__(self):
        self.required_datafields = {"PartType0/Coordinates", "PartType0/SmoothingLength"}
        self.plotlabel = ""
        self.colormap = "viridis"


class SurfaceDensity(RenderMap):
    # plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"
    def __init__(self):
        super().__init__()
        self.plotlabel = r"$\Sigma_{\rm gas}\,\left(M_\odot\,\rm pc^{-2}\right)$"
        self.required_datafields.update({"PartType0/Masses"})

    def render(self, pdata: dict, meshoid: Meshoid, mapargs: dict):
        return meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)


class VelocityDispersion(RenderMap):
    def __init__(self):
        super().__init__()
        self.plotlabel = r"$\sigma_{\rm 1D}\,\left(\rm km\,s^{-1}\right)$"
        self.colormap = "magma"
        self.required_datafields.update({"PartType0/Masses", "PartType0/Velocities"})

    def render(self, pdata: dict, meshoid: Meshoid, mapargs: dict):
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
        mvSqr_map = (
            meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2] ** 2, **mapargs)
            / sigma
        )
        mvmap = (
            meshoid.SurfaceDensity(pdata["PartType0/Masses"] * pdata["PartType0/Velocities"][:, 2], **mapargs) / sigma
        )
        return np.clip(mvSqr_map - mvmap**2, 0, 1e100)


class MassWeightedTemperature(RenderMap):
    def __init__(self):
        super().__init__()
        self.plotlabel = r"$\rm \langle T \rangle_{\rm M} \,\left(K\right)$"
        self.colormap = "plasma"
        self.required_datafields.update({"PartType0/Masses", "PartType0/Temperature"})

    def render(self, pdata: dict, meshoid: Meshoid, mapargs: dict):
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
        Tsigma = meshoid.SurfaceDensity(pdata["PartType0/Temperature"], **mapargs)
        return Tsigma / sigma


class AlfvenSpeed(RenderMap):
    def __init__(self):
        super().__init__()
        self.plotlabel = r"$v_{\rm A,RMS}\,\left(\rm km\,s^{-1}\right)$"
        self.colormap = "RdYlBu"
        self.required_datafields.update({"PartType0/Masses", "PartType0/MagneticField"})

    def render(self, pdata: dict, meshoid: Meshoid, mapargs: dict):
        Bsqr = np.sum(pdata["PartType0/MagneticField"] ** 2, axis=1)
        tesla2_to_egydens = 5.88e24  # 1/(2 mu0) in our stupid units
        sigma_Emag = meshoid.Projection(Bsqr * tesla2_to_egydens, **mapargs)
        sigma = meshoid.SurfaceDensity(pdata["PartType0/Masses"], **mapargs)
        return np.sqrt(2 * sigma_Emag / sigma)
