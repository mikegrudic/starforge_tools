"""Routines for feedback rates from individual stars in the STARFORGE model"""

from astropy import units as u, constants as c
import numpy as np
from .special_functions import planck_integral


def luminosity_MS(mass):
    """Main-sequence luminosity from Tout 1996MNRAS.281..257T.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.

    Returns
    -------
    ndarray
        Luminosity in solar luminosities.
    """
    lum_ms = (0.39704170 * np.power(mass, 5.5) + 8.52762600 * np.power(mass, 11.0)) / (
        0.00025546
        + np.power(mass, 3.0)
        + 5.43288900 * np.power(mass, 5.0)
        + 5.56357900 * np.power(mass, 7.0)
        + 0.78866060 * np.power(mass, 8.0)
        + 0.00586685 * np.power(mass, 9.5)
    )
    lum_ms = np.atleast_1d(lum_ms)
    lum_ms[np.isnan(lum_ms)] = 0.0
    return np.float64(lum_ms)


def radius_MS(mass):
    """Main-sequence radius from Tout 1996MNRAS.281..257T.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.

    Returns
    -------
    ndarray
        Radius in solar radii.
    """
    radius_ms = (
        1.71535900 * np.power(mass, 2.5)
        + 6.59778800 * np.power(mass, 6.5)
        + 10.08855000 * np.power(mass, 11.0)
        + 1.01249500 * np.power(mass, 19.0)
        + 0.07490166 * np.power(mass, 19.5)
    ) / (
        0.01077422
        + 3.08223400 * np.power(mass, 2.0)
        + 17.84778000 * np.power(mass, 8.5)
        + np.power(mass, 18.5)
        + 0.00022582 * np.power(mass, 19.5)
    )
    radius_ms = np.atleast_1d(radius_ms)
    radius_ms[np.isnan(radius_ms)] = 0.0
    return np.float64(radius_ms)


VESC_FAC = np.sqrt(2 * c.G * c.M_sun / c.R_sun).to(u.km / u.s).value


def v_escape(m_solar, r_solar=None):
    """Surface escape speed sqrt(2GM/R).

    Parameters
    ----------
    m_solar : array_like
        Stellar mass in solar masses.
    r_solar : array_like, optional
        Stellar radius in solar radii. Defaults to the main-sequence radius
        from `radius_MS`.

    Returns
    -------
    ndarray
        Escape speed in km/s.
    """
    if r_solar is None:
        r_solar = radius_MS(m_solar)
    return VESC_FAC * np.sqrt(m_solar / r_solar)


def effective_temperature(mass=None, lum=None, radius=None):
    """Stellar effective temperature from the Stefan-Boltzmann law.

    Parameters
    ----------
    mass : array_like, optional
        Stellar mass in solar masses. If provided without `lum` and `radius`,
        main-sequence values are used.
    lum : array_like, optional
        Luminosity in solar luminosities.
    radius : array_like, optional
        Radius in solar radii.

    Returns
    -------
    ndarray
        Effective temperature in Kelvin.
    """
    if mass is not None and ((lum is None) or (radius is None)):
        lum, radius = luminosity_MS(mass), radius_MS(mass)
    return 5814.33 * (lum / radius**2) ** 0.25


def vwind_over_vesc(T_eff):
    """Stellar wind velocity in units of the escape speed (Lamers 1995).

    Returns 0.7 for T_eff < 12500 K, 1.3 for T_eff < 21000 K, and 2.6
    otherwise, reflecting the bistability jumps in line-driven winds.

    Parameters
    ----------
    T_eff : array_like
        Effective temperature in Kelvin.

    Returns
    -------
    ndarray
        Wind speed as a multiple of the surface escape speed.
    """
    T_eff = np.asarray(T_eff)
    return np.where(T_eff < 1.25e4, 0.7, np.where(T_eff < 2.1e4, 1.3, 2.6))


def vwind(mass, lum=None, radius=None):
    """Stellar wind speed.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.
    lum : array_like, optional
        Luminosity in solar luminosities. Defaults to main-sequence value.
    radius : array_like, optional
        Radius in solar radii. Defaults to main-sequence value.

    Returns
    -------
    ndarray
        Wind speed in km/s.
    """
    T_eff = effective_temperature(mass, lum, radius)
    return vwind_over_vesc(T_eff) * v_escape(mass, radius)


def mdot_vms(mass, lum=None, radius=None, Z=1.0):
    """Wind mass-loss rate for very massive stars (VMS) per Sabhahit arXiv:2205.09125 Eq. 13.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.
    lum : array_like, optional
        Luminosity in solar luminosities. Defaults to main-sequence value.
    radius : array_like, optional
        Radius in solar radii. Defaults to main-sequence value.
    Z : float, optional
        Metallicity in solar units. Default is 1.0.

    Returns
    -------
    ndarray
        Mass-loss rate in solar masses per year.
    """
    if mass is not None and ((lum is None) or (radius is None)):
        lum, radius = luminosity_MS(mass), radius_MS(mass)

    T_eff = effective_temperature(mass, lum, radius)
    logmdot_wind_high = (
        -8.445
        + 4.77 * np.log10(lum.clip(1e-10) / 1e5)
        - 3.99 * np.log10(mass / 30)
        - 1.226 * np.log10(vwind_over_vesc(T_eff) / 2)
        + 0.761 * np.log10(Z)
    )
    return 10**logmdot_wind_high


def wind_mdot(mass=None, lum=None, Z_solar=1.0, vms=True):
    """Main-sequence wind mass-loss rate used in the STARFORGE model (SINGLE_STAR_FB_WINDS == 2).

    Implements the "de Jager / 3" prescription from Smith (2014) with a
    weak-wind limiter (stellar_evolution.cc:1015,1017). Metallicity scaling
    (Z^0.69) applies only to the de Jager term, not the weak-wind limiter.
    The VMS floor from `mdot_vms` (Sabhahit arXiv:2205.09125 Eq. 13) is
    applied by default (SINGLE_STAR_FB_WINDS & 2); set `vms=False` to match
    SINGLE_STAR_FB_WINDS == 0.

    Parameters
    ----------
    mass : array_like, optional
        Stellar mass in solar masses.
    lum : array_like, optional
        Luminosity in solar luminosities. Computed from `mass` if not given.
    Z_solar : float, optional
        Metallicity in solar units. Default is 1.0.
    vms : bool, optional
        If True, apply the VMS mass-loss floor from `mdot_vms`. Default is
        True (matches Gizmo's SINGLE_STAR_FB_WINDS == 2).

    Returns
    -------
    ndarray
        Mass-loss rate in solar masses per year.
    """
    if lum is None:
        lum = luminosity_MS(mass)
    mdot_dejager = 10**-15.0 * lum**1.5 * Z_solar**0.69  # de Jager/"3", Smith 2014
    mdot_weak = 10**-22.15 * lum**2.9  # weak-wind limiter
    mdot = np.minimum(mdot_dejager, mdot_weak)
    if vms:
        radius = radius_MS(mass)
        mdot_hi_2 = mdot_vms(mass, lum, radius, Z_solar)
        mdot = np.maximum(mdot, mdot_hi_2)
    return mdot


def Q_ionizing(mass=None, lum=None, radius=None, energy_eV=13.6):
    """Ionizing photon emission rate for a blackbody spectrum, computed to machine precision.

    Parameters
    ----------
    mass : array_like, optional
        Stellar mass in solar masses. Used to infer `lum` and `radius` if not
        provided.
    lum : array_like, optional
        Luminosity in solar luminosities.
    radius : array_like, optional
        Radius in solar radii.
    energy_eV : float, optional
        Ionization threshold in eV. Default is 13.6 eV (hydrogen).

    Returns
    -------
    ndarray
        Ionizing photon emission rate in photons per second.
    """
    if mass is not None and ((lum is None) or (radius is None)):
        lum, radius = luminosity_MS(mass).clip(1e-10), radius_MS(mass)

    T_eff = effective_temperature(mass, lum, radius).clip(1e-10)
    k_B = 8.617e-5  # in eV/K
    x1 = energy_eV / (k_B * T_eff)
    Lsun_cgs = 2.389e45
    planck_integral_fac = 0.37020884510871604  # ratio of integral of x^2/(exp(x)-1) over that of x^3/(exp(x) - 1)
    return np.float64(lum) * Lsun_cgs / (k_B * T_eff) * planck_integral(x1, np.inf, 2) * planck_integral_fac


def Q_ionizing_approx(mass, energy_eV=13.6):
    """Ionizing photon emission rate for a blackbody spectrum, accurate to within ~5%.

    Faster than `Q_ionizing` due to the polynomial approximation used for the
    Planck integral. Prefer `Q_ionizing` when accuracy matters.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.
    energy_eV : float, optional
        Ionization threshold in eV. Default is 13.6 eV (hydrogen).

    Returns
    -------
    ndarray
        Ionizing photon emission rate in photons per second.
    """
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = energy_eV / (k_B * T_eff)
    ionizing_frac = ionizing_frac_approx(x1)
    return ionizing_frac * L * 1.7e44 / (1 + 3 / x1 - 2 * (1 + x1) / (2 + x1 * (2 + x1)))


def ionizing_frac_approx(x1):
    """Fraction of blackbody luminosity emitted above E = x1 * k_B * T_eff.

    Uses a low-x polynomial and a high-x exponential approximation, matched
    at x1 = 2.71.

    Parameters
    ----------
    x1 : array_like
        Dimensionless energy threshold E / (k_B * T_eff).

    Returns
    -------
    ndarray
        Fraction of total luminosity above the threshold.
    """
    result = np.empty_like(x1)
    result[x1 < 2.710528524106676] = (
        1
        - ((131.4045728599595 * x1 * x1 * x1) / (2560.0 + x1 * (960.0 + x1 * (232.0 + 39.0 * x1))))[
            x1 < 2.710528524106676
        ]
    )
    # approximation of integral of Planck function from 0 to x1, valid for x1 << 1 \
    result[x1 >= 2.710528524106676] = ((0.15398973382026504 * (6.0 + x1 * (6.0 + x1 * (3.0 + x1)))) * np.exp(-x1))[
        x1 >= 2.710528524106676
    ]
    # approximation of Planck integral for large x
    return result


def lum_ionizing(mass):
    """Luminosity in hydrogen-ionizing photons (E > 13.6 eV) assuming a blackbody spectrum.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.

    Returns
    -------
    ndarray
        Ionizing luminosity in solar luminosities.
    """
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = 13.6 / (k_B * T_eff)
    ionizing_frac = planck_integral(x1, np.inf, 3)
    return ionizing_frac * L


def lum_band(mass, E1, E2=np.inf):
    """Luminosity in a specified photon energy band, assuming a blackbody spectrum.

    Parameters
    ----------
    mass : array_like
        Stellar mass in solar masses.
    E1 : float
        Lower energy bound in eV.
    E2 : float, optional
        Upper energy bound in eV. Default is infinity.

    Returns
    -------
    ndarray
        Band luminosity in solar luminosities.
    """
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = E1 / (k_B * T_eff)
    x2 = E2 / (k_B * T_eff)
    frac = planck_integral(x1, x2, 3)
    return np.abs(frac) * L
