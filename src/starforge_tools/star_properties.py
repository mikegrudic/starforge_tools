"""Routines for feedback rates from individual stars in the STARFORGE model"""

from astropy import units as u, constants as c
import numpy as np
from .special_functions import planck_integral


def _strip(value, unit):
    """Return plain numerical value in `unit`; assumes `unit` if given a plain number."""
    if isinstance(value, u.Quantity):
        return value.to(unit).value
    return np.asarray(value, dtype=float)


def luminosity_MS(mass):
    """Main-sequence luminosity from Tout 1996MNRAS.281..257T.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.

    Returns
    -------
    Quantity
        Luminosity in solar luminosities.
    """
    mass = _strip(mass, u.M_sun)
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
    return np.float64(lum_ms) * u.L_sun


def radius_MS(mass):
    """Main-sequence radius from Tout 1996MNRAS.281..257T.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.

    Returns
    -------
    Quantity
        Radius in solar radii.
    """
    mass = _strip(mass, u.M_sun)
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
    return np.float64(radius_ms) * u.R_sun


def stellar_lifetime(mass):
    """Main-sequence lifetime (stellar_evolution.cc:1137).

    Uses the ZAMS luminosity from `luminosity_MS` as in the STARFORGE
    protostellar evolution model (SINGLE_STAR_STARFORGE_PROTOSTELLAR_EVOLUTION
    == 2), rather than the instantaneous luminosity.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar ZAMS mass. Assumed solar masses if not a Quantity.

    Returns
    -------
    Quantity
        Lifetime in Gyr. Gives ~10 Gyr for solar-type stars, ~40 Myr for
        8 M☉, and asymptotes to ~3.7 Myr at very high mass.
    """
    mass = _strip(mass, u.M_sun)
    lum = _strip(luminosity_MS(mass), u.L_sun)
    return (9.6 * (mass / lum) + 0.0034) * u.Gyr


VESC_FAC = np.sqrt(2 * c.G * c.M_sun / c.R_sun).to(u.km / u.s).value


def v_escape(m_solar, r_solar=None):
    """Surface escape speed sqrt(2GM/R).

    Parameters
    ----------
    m_solar : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.
    r_solar : array_like or Quantity, optional
        Stellar radius. Assumed solar radii if not a Quantity. Defaults to
        the main-sequence radius from `radius_MS`.

    Returns
    -------
    Quantity
        Escape speed in km/s.
    """
    mass = _strip(m_solar, u.M_sun)
    radius = _strip(r_solar, u.R_sun) if r_solar is not None else _strip(radius_MS(mass), u.R_sun)
    return VESC_FAC * np.sqrt(mass / radius) * (u.km / u.s)


def effective_temperature(mass=None, lum=None, radius=None):
    """Stellar effective temperature from the Stefan-Boltzmann law.

    Parameters
    ----------
    mass : array_like or Quantity, optional
        Stellar mass. Assumed solar masses if not a Quantity. If provided
        without `lum` and `radius`, main-sequence values are used.
    lum : array_like or Quantity, optional
        Luminosity. Assumed solar luminosities if not a Quantity.
    radius : array_like or Quantity, optional
        Radius. Assumed solar radii if not a Quantity.

    Returns
    -------
    Quantity
        Effective temperature in Kelvin.
    """
    if mass is not None:
        mass = _strip(mass, u.M_sun)
    if lum is not None:
        lum = _strip(lum, u.L_sun)
    if radius is not None:
        radius = _strip(radius, u.R_sun)
    if mass is not None and (lum is None or radius is None):
        lum = _strip(luminosity_MS(mass), u.L_sun)
        radius = _strip(radius_MS(mass), u.R_sun)
    return 5814.33 * (lum / radius**2) ** 0.25 * u.K


def vwind_over_vesc(T_eff):
    """Stellar wind velocity in units of the escape speed (Lamers 1995).

    Returns 0.7 for T_eff < 12500 K, 1.3 for T_eff < 21000 K, and 2.6
    otherwise, reflecting the bistability jumps in line-driven winds.

    Parameters
    ----------
    T_eff : array_like or Quantity
        Effective temperature. Assumed Kelvin if not a Quantity.

    Returns
    -------
    ndarray
        Wind speed as a multiple of the surface escape speed (dimensionless).
    """
    T_K = _strip(T_eff, u.K)
    return np.where(T_K < 1.25e4, 0.7, np.where(T_K < 2.1e4, 1.3, 2.6))


def vwind(mass, lum=None, radius=None):
    """Stellar wind speed.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.
    lum : array_like or Quantity, optional
        Luminosity. Assumed solar luminosities if not a Quantity. Defaults
        to main-sequence value.
    radius : array_like or Quantity, optional
        Radius. Assumed solar radii if not a Quantity. Defaults to
        main-sequence value.

    Returns
    -------
    Quantity
        Wind speed in km/s.
    """
    T_eff = effective_temperature(mass, lum, radius)
    return vwind_over_vesc(T_eff) * v_escape(mass, radius)


def mdot_vms(mass, lum=None, radius=None, Z=1.0):
    """Wind mass-loss rate for very massive stars (VMS) per Sabhahit arXiv:2205.09125 Eq. 13.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.
    lum : array_like or Quantity, optional
        Luminosity. Assumed solar luminosities if not a Quantity. Defaults
        to main-sequence value.
    radius : array_like or Quantity, optional
        Radius. Assumed solar radii if not a Quantity. Defaults to
        main-sequence value.
    Z : float, optional
        Metallicity in solar units. Default is 1.0.

    Returns
    -------
    Quantity
        Mass-loss rate in solar masses per year.
    """
    mass = _strip(mass, u.M_sun)
    lum = _strip(lum if lum is not None else luminosity_MS(mass), u.L_sun)
    radius = _strip(radius if radius is not None else radius_MS(mass), u.R_sun)
    T_K = _strip(effective_temperature(None, lum, radius), u.K)
    logmdot = (
        -8.445
        + 4.77 * np.log10(np.clip(lum, 1e-10, None) / 1e5)
        - 3.99 * np.log10(mass / 30)
        - 1.226 * np.log10(vwind_over_vesc(T_K) / 2)
        + 0.761 * np.log10(Z)
    )
    return 10**logmdot * (u.M_sun / u.yr)


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
    mass : array_like or Quantity, optional
        Stellar mass. Assumed solar masses if not a Quantity.
    lum : array_like or Quantity, optional
        Luminosity. Assumed solar luminosities if not a Quantity. Computed
        from `mass` if not given.
    Z_solar : float, optional
        Metallicity in solar units. Default is 1.0.
    vms : bool, optional
        If True, apply the VMS mass-loss floor from `mdot_vms`. Default is
        True (matches Gizmo's SINGLE_STAR_FB_WINDS == 2).

    Returns
    -------
    Quantity
        Mass-loss rate in solar masses per year.
    """
    if mass is not None:
        mass = _strip(mass, u.M_sun)
    lum = _strip(lum if lum is not None else luminosity_MS(mass), u.L_sun)
    mdot_dejager = 10**-15.0 * lum**1.5 * Z_solar**0.69  # de Jager/"3", Smith 2014
    mdot_weak = 10**-22.15 * lum**2.9  # weak-wind limiter
    mdot = np.minimum(mdot_dejager, mdot_weak)
    if vms:
        mdot_hi_2 = _strip(mdot_vms(mass, lum, None, Z_solar), u.M_sun / u.yr)
        mdot = np.maximum(mdot, mdot_hi_2)
    return mdot * (u.M_sun / u.yr)


def Q_ionizing(mass=None, lum=None, radius=None, energy_eV=13.6):
    """Ionizing photon emission rate for a blackbody spectrum, computed to machine precision.

    Parameters
    ----------
    mass : array_like or Quantity, optional
        Stellar mass. Assumed solar masses if not a Quantity. Used to infer
        `lum` and `radius` if not provided.
    lum : array_like or Quantity, optional
        Luminosity. Assumed solar luminosities if not a Quantity.
    radius : array_like or Quantity, optional
        Radius. Assumed solar radii if not a Quantity.
    energy_eV : float or Quantity, optional
        Ionization threshold. Assumed eV if not a Quantity. Default is
        13.6 eV (hydrogen).

    Returns
    -------
    Quantity
        Ionizing photon emission rate in photons per second (s^-1).
    """
    if mass is not None:
        mass = _strip(mass, u.M_sun)
    if lum is not None:
        lum = _strip(lum, u.L_sun)
    if radius is not None:
        radius = _strip(radius, u.R_sun)
    if mass is not None and (lum is None or radius is None):
        lum = np.clip(_strip(luminosity_MS(mass), u.L_sun), 1e-10, None)
        radius = _strip(radius_MS(mass), u.R_sun)
    energy = _strip(energy_eV, u.eV)
    # nan_to_num: clip alone passes NaN through (e.g. L=0 and R=0 give T=0/0)
    T_K = np.clip(np.nan_to_num(_strip(effective_temperature(None, lum, radius), u.K)), 1e-10, None)
    k_B = 8.617e-5  # eV/K
    x1 = energy / (k_B * T_K)
    Lsun_cgs = 2.389e45
    planck_integral_fac = 0.37020884510871604  # ratio of integral of x^2/(exp(x)-1) over that of x^3/(exp(x) - 1)
    return np.float64(lum) * Lsun_cgs / (k_B * T_K) * planck_integral(x1, np.inf, 2) * planck_integral_fac / u.s


def Q_ionizing_approx(mass, energy_eV=13.6):
    """Ionizing photon emission rate for a blackbody spectrum, accurate to within ~5%.

    Faster than `Q_ionizing` due to the polynomial approximation used for the
    Planck integral. Prefer `Q_ionizing` when accuracy matters.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.
    energy_eV : float or Quantity, optional
        Ionization threshold. Assumed eV if not a Quantity. Default is
        13.6 eV (hydrogen).

    Returns
    -------
    Quantity
        Ionizing photon emission rate in photons per second (s^-1).
    """
    mass = _strip(mass, u.M_sun)
    L = _strip(luminosity_MS(mass), u.L_sun)
    R = _strip(radius_MS(mass), u.R_sun)
    T_K = np.clip(np.nan_to_num(_strip(effective_temperature(None, L, R), u.K)), 1e-10, None)
    energy = _strip(energy_eV, u.eV)
    k_B = 8.617e-5
    x1 = energy / (k_B * T_K)
    ionizing_frac = ionizing_frac_approx(x1)
    return ionizing_frac * L * 1.7e44 / (1 + 3 / x1 - 2 * (1 + x1) / (2 + x1 * (2 + x1))) / u.s


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
    # full_like(nan) not empty_like: a NaN x1 matches neither branch mask below,
    # which would leave uninitialized memory in the result
    result = np.full_like(np.asarray(x1, dtype=np.float64), np.nan)
    result[x1 < 2.710528524106676] = (
        1
        - ((131.4045728599595 * x1 * x1 * x1) / (2560.0 + x1 * (960.0 + x1 * (232.0 + 39.0 * x1))))[
            x1 < 2.710528524106676
        ]
    )
    result[x1 >= 2.710528524106676] = ((0.15398973382026504 * (6.0 + x1 * (6.0 + x1 * (3.0 + x1)))) * np.exp(-x1))[
        x1 >= 2.710528524106676
    ]
    return result


def lum_ionizing(mass):
    """Luminosity in hydrogen-ionizing photons (E > 13.6 eV) assuming a blackbody spectrum.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.

    Returns
    -------
    Quantity
        Ionizing luminosity in solar luminosities.
    """
    mass = _strip(mass, u.M_sun)
    L = _strip(luminosity_MS(mass), u.L_sun)
    R = _strip(radius_MS(mass), u.R_sun)
    T_K = np.clip(np.nan_to_num(_strip(effective_temperature(None, L, R), u.K)), 1e-10, None)
    k_B = 8.617e-5
    x1 = 13.6 / (k_B * T_K)
    return planck_integral(x1, np.inf, 3) * L * u.L_sun


def lum_band(mass, E1, E2=np.inf):
    """Luminosity in a specified photon energy band, assuming a blackbody spectrum.

    Parameters
    ----------
    mass : array_like or Quantity
        Stellar mass. Assumed solar masses if not a Quantity.
    E1 : float or Quantity
        Lower energy bound. Assumed eV if not a Quantity.
    E2 : float or Quantity, optional
        Upper energy bound. Assumed eV if not a Quantity. Default is infinity.

    Returns
    -------
    Quantity
        Band luminosity in solar luminosities.
    """
    mass = _strip(mass, u.M_sun)
    L = _strip(luminosity_MS(mass), u.L_sun)
    R = _strip(radius_MS(mass), u.R_sun)
    T_K = np.clip(np.nan_to_num(_strip(effective_temperature(None, L, R), u.K)), 1e-10, None)
    k_B = 8.617e-5
    x1 = _strip(E1, u.eV) / (k_B * T_K)
    x2 = _strip(E2, u.eV) / (k_B * T_K)
    return np.abs(planck_integral(x1, x2, 3)) * L * u.L_sun
