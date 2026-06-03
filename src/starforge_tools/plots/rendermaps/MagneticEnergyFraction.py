"""Magnetic energy fraction map: E_mag / (E_kin + E_therm) integrated along the line of sight"""

from meshoid import Meshoid
import numpy as np
from .rendermap import RenderMap


class MagneticEnergyFraction(RenderMap):
    plotlabel = r"$E_{\rm mag} / (E_{\rm kin} + E_{\rm thermal})$"
    required_datafields = (
        "PartType0/Coordinates",
        "PartType0/Masses",
        "PartType0/SmoothingLength",
        "PartType0/MagneticField",
        "PartType0/Velocities",
        "PartType0/InternalEnergy",
    )
    colormap = "RdBu_r"
    cmap_default_limits = (1e-2, 1e2)

    @staticmethod
    def render(pdata: dict, meshoid: Meshoid, mapargs: dict):
        masses = pdata["PartType0/Masses"]
        Bsqr = np.sum(pdata["PartType0/MagneticField"] ** 2, axis=1)
        vsqr = np.sum(pdata["PartType0/Velocities"] ** 2, axis=1)
        u = pdata["PartType0/InternalEnergy"]

        # Magnetic side: tesla2_to_egydens converts B^2 [T^2] -> energy density in
        # M_sun (km/s)^2 / pc^3 (matches AlfvenSpeed). Snapshot B is in Tesla
        # numerically (per io.get_snapshot_units: 1 code unit B = 1e4 G = 1 T).
        tesla2_to_egydens = 5.88e18  # 1/(2 mu0) in M_sun (km/s)^2 / pc^3 per T^2
        sigma_Emag = meshoid.Projection(Bsqr * tesla2_to_egydens, **mapargs)

        # Kinetic / thermal side: snapshot velocity unit may not be km/s.
        # Convert specific-energy quantities from (snapshot v)^2 to (km/s)^2
        # using UnitVelocity_In_CGS (cm/s); km/s = 1e5 cm/s.
        v_to_kms_sq = (pdata["Header"]["UnitVelocity_In_CGS"] / 1e5) ** 2
        sigma_Ekin = meshoid.SurfaceDensity(0.5 * masses * vsqr * v_to_kms_sq, **mapargs)
        sigma_Etherm = meshoid.SurfaceDensity(masses * u * v_to_kms_sq, **mapargs)

        return sigma_Emag / (sigma_Ekin + sigma_Etherm)
