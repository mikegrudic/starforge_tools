# This code is copied almost verbatim from Phil Hopkins' GIZMO analysis scripts: https://bitbucket.org/phopkins/pfh_python/
## return temperature from internal energy, helium fraction, electron abundance, total metallicity, and density
def get_temperature(
    internal_egy_code,
    helium_mass_fraction,
    electron_abundance,
    total_metallicity,
    mass_density,
    f_neutral=np.zeros(0),
    f_molec=np.zeros(0),
    key="Temperature",
):
    """Return estimated gas temperature, given code-units internal energy, helium
    mass fraction, and electron abundance (number of free electrons per H nucleus).
    this will use a simply approximation to the iterative in-code solution for
    molecular gas, total metallicity, etc. so does not perfectly agree
    (but it is a few lines, as opposed to hundreds)


    """
    internal_egy_cgs = internal_egy_code * 1.0e4
    gamma_EOS = 5.0 / 3.0
    kB = 1.38e-16
    m_proton = 1.67e-24
    X0 = 0.76
    total_metallicity[(total_metallicity > 0.25)] = 0.25
    helium_mass_fraction[(helium_mass_fraction > 0.35)] = 0.35
    y_helium = helium_mass_fraction / (4.0 * (1.0 - helium_mass_fraction))
    X_hydrogen = 1.0 - (helium_mass_fraction + total_metallicity)
    T_mol = 100.0 * (mass_density * 0.404621)
    T_mol[(T_mol > 8000.0)] = 8000.0
    A0 = m_proton * (gamma_EOS - 1.0) * internal_egy_cgs / kB
    mu = (1.0 + 4.0 * y_helium) / (1.0 + y_helium + electron_abundance)
    X = X_hydrogen
    Y = helium_mass_fraction
    Z = total_metallicity
    nel = electron_abundance

    if np.array(f_molec).size > 0:
        fmol = 1.0 * f_molec
        fH = X_hydrogen
        f = fmol
        xe = nel
        f_mono = fH * (xe + 1.0 - f) + (1.0 - fH) / 4.0
        f_di = fH * f / 2.0
        gamma_mono = 5.0 / 3.0
        gamma_di = 7.0 / 5.0
        gamma_eff = 1.0 + (f_mono + f_di) / (
            f_mono / (gamma_mono - 1.0) + f_di / (gamma_di - 1.0)
        )
        A0 = m_proton * (gamma_eff - 1.0) * internal_egy_cgs / kB
    else:
        fmol = 0
        if 2 == 2:
            for i in range(3):
                mu = 1.0 / (
                    X * (1.0 - 0.5 * fmol)
                    + Y / 4.0
                    + nel * X0
                    + Z / (16.0 + 12.0 * fmol)
                )
                T = mu * A0 / T_mol
                fmol = 1.0 / (1.0 + T * T)
    mu = 1.0 / (X * (1.0 - 0.5 * fmol) + Y / 4.0 + nel * X0 + Z / (16.0 + 12.0 * fmol))
    T = mu * A0
    if "Temp" in key or "temp" in key:
        return T
    if "Weight" in key or "weight" in key:
        return mu
    if "Number" in key or "number" in key:
        return mass_density * 40.4621 / mu
    return T
