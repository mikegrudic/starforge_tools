"""Main-sequence stellar properties derived from Tout 1996 formulae and STARFORGE wind and feedback prescriptions"""
import numpy as np
from special_functions import planck2_integral, planck3_integral


def luminosity_MS(mass):
    """Fit of main-sequence luminosity as a function of main-sequence mass from Tout 1996MNRAS.281..257T, in solar units"""
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
    return lum_ms


def radius_MS(mass):
    """Fit of main-sequence radius as a function of main-sequence mass from
    Tout 1996MNRAS.281..257T, in solar units"""
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
    return radius_ms


def effective_temperature(mass=None, lum=None, radius=None):
    """Effective temperature as a function of mass (assuming main sequence)
    or optionally luminosity and radius"""
    if mass is not None and ((lum is None) or (radius is None)):
        lum, radius = luminosity_MS(mass), radius_MS(mass)
    # print(mass, lum, radius)
    return 5780 * (lum / radius**2) ** 0.25


def wind_mdot_MS(mass=None, lum=None, Z_solar=1.0):
    """Returns the main-sequence wind mass loss rate in solar mass per year,
    following 2022MNRAS.512..216G"""
    if lum is None:
        lum = luminosity_MS(mass)
    mdot_hi = 10**-22.2 * lum**2.9  # weak wind
    mdot_lower = 10**-15.0 * lum**1.5  # O stars
    return np.min([mdot_hi, mdot_lower], axis=0) * Z_solar**0.7


def Q_ionizing(mass, energy_eV=13.6):
    """Number of photons with energy > energy_eV emitted per second, assuming blackbody spectrum"""
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = energy_eV / (k_B * T_eff)
    return L * 2.389e45 / (k_B * T_eff) * planck2_integral(x1, np.inf)


def Q_ionizing_approx(mass, energy_eV=13.6):
    """Number of photons with energy > energy_eV emitted per second, assuming blackbody spectrum"""
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = energy_eV / (k_B * T_eff)
    ionizing_frac = ionizing_frac_approx(x1)
    return (
        ionizing_frac * L * 1.7e44 / (1 + 3 / x1 - 2 * (1 + x1) / (2 + x1 * (2 + x1)))
    )


def ionizing_frac_approx(x1):
    """Approximates fraction of luminosity above E = x * k_B * T_eff"""
    result = np.empty_like(x1)
    result[x1 < 2.710528524106676] = (
        1
        - (
            (131.4045728599595 * x1 * x1 * x1)
            / (2560.0 + x1 * (960.0 + x1 * (232.0 + 39.0 * x1)))
        )[x1 < 2.710528524106676]
    )
    # approximation of integral of Planck function from 0 to x1, valid for x1 << 1 \
    result[x1 >= 2.710528524106676] = (
        (0.15398973382026504 * (6.0 + x1 * (6.0 + x1 * (3.0 + x1)))) * np.exp(-x1)
    )[x1 >= 2.710528524106676]
    # approximation of Planck integral for large x
    return result


def lum_ionizing(mass):
    """Luminosity of photons with energy > energy_eV emitted per second, assuming blackbody spectrum"""
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = 13.6 / (k_B * T_eff)
    ionizing_frac = planck3_integral(x1, np.inf)
    return ionizing_frac * L


def lum_band(mass, E1, E2=np.inf):
    """Luminosity in specified energy band in eV, assuming blackbody spectrum"""
    L, R = luminosity_MS(mass), radius_MS(mass)
    T_eff = effective_temperature(mass, L, R)
    k_B = 8.617e-5
    x1 = E1 / (k_B * T_eff)
    x2 = E2 / (k_B * T_eff)
    frac = planck3_integral(x1, x2)
    return np.abs(frac) * L
