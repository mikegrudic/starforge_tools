"""Routine for computing non-ideal MHD coefficients from simulation data in post-processing"""

import numpy as np

GRAVITY_G_CGS = 6.672e-8
SOLAR_MASS_CGS = 1.989e33
SOLAR_LUM_CGS = 3.826e33
SOLAR_RADIUS_CGS = 6.957e10
BOLTZMANN_CGS = 1.38066e-16
C_LIGHT_CGS = 2.9979e10
PROTONMASS_CGS = 1.6726e-24
ELECTRONMASS_CGS = 9.10953e-28
THOMPSON_CX_CGS = 6.65245e-25
ELECTRONCHARGE_CGS = 4.8032e-10
SECONDS_PER_YEAR = 3.155e7
HUBBLE_H100_CGS = 3.2407789e-18
ELECTRONVOLT_IN_ERGS = 1.60217733e-12
HYDROGEN_MASSFRAC = 0.76

UNIT_B_IN_GAUSS = 1e4  # specific to STARFORGE convention, where snapshots are in tesla


def nonideal_mhd_coefficients(F, a_grain_micron=0.1):
    """Given a snapshot file, return the non-ideal MHD resistivies in CGS

    Parameters
    ----------
    F : h5py.File
        an h5py.File instance pointing to the snapshot
    a_grain_micron: float, optional
        The assumed grain size in microns

    Returns
    -------
    eta_O, eta_H, eta A: arrays containing non-ideal resistivities in cm^2/s
    """
    # calculations below follow Wardle 2007 and Keith & Wardle 2014, for the equation sets */
    # molecular H2, +He with solar mass fractions and metals
    mean_molecular_weight = 2.38
    f_dustgas = 0.01  # effective size of grains that matter at these densities
    m_ion = 24.3  # Mg dominates ions in dense gas [where this is relevant] this is ion mass in units of proton mass
    zeta_cr = 1e-17  # cosmic ray ionization rate (fixed as constant for non-CR runs)
    # appropriate dust-to-metals ratio
    f_dustgas = 0.5 * F["PartType0/Metallicity"][:, 0]
    # will use appropriate EOS to estimate temperature
    temperature = F["PartType0/Temperature"][:]  # .clip(10, 10)
    # now everything should be fully-determined (given the inputs above and the known properties of the gas) #
    m_neutral = mean_molecular_weight  # in units of the proton mass
    ag01 = a_grain_micron / 0.1
    m_grain = 7.51e9 * ag01 * ag01 * ag01  # grain mass [internal density =3 g/cm^3]
    M_unit = F["Header"].attrs["UnitMass_In_CGS"]
    L_unit = F["Header"].attrs["UnitLength_In_CGS"]
    # B_unit = F["Header"].attrs["Internal_UnitB_In_Gauss"]
    # print(B_unit)
    V_unit = F["Header"].attrs["UnitVelocity_In_CGS"]
    UNIT_PRESSURE_IN_CGS = M_unit / L_unit**3 * V_unit**2
    # UNIT_B_IN_GAUSS = np.sqrt(4.0 * np.pi * UNIT_PRESSURE_IN_CGS)

    UNIT_DENSITY_IN_CGS = M_unit / L_unit**3
    rho = F["PartType0/Density"][:] * (F["Header"].attrs["HubbleParam"] ** -3) * UNIT_DENSITY_IN_CGS
    n_eff = np.float64(rho) / PROTONMASS_CGS  # density in cgs
    # calculate ionization fraction in dense gas use rate coefficients k to estimate grain charge
    # prefactor for rate coefficient for electron-grain collisions

    #
    ngr_ngas = (m_neutral / m_grain) * f_dustgas  # number of grains per neutral
    # # e*e/(a_grain*k_boltzmann*T): Z_grain = psi/psi_prefac where psi is constant determines charge
    psi_prefac = 167.1 / (ag01 * temperature)
    # # coefficient for equation that determines Z_grain
    # # psi solves the equation: psi = alpha * (exp[psi] - y/(1+psi)) where y=np.sqrt(m_ion/m_electron) note the solution for small alpha is independent of m_ion, only large alpha
    # #   (where the non-ideal effects are weak, generally) produces a difference: at very high-T, appropriate m_ion should be hydrogen+helium, but in this limit our cooling
    # #    routines will already correctly determine the ionization states. so we can safely adopt Mg as our ion of consideration
    y = np.sqrt(m_ion * PROTONMASS_CGS / ELECTRONMASS_CGS)

    ####### This stuff makes various assumptions to determine ionization from zeta_cr
    # k0 = 1.95e-4 * ag01 * ag01 * np.sqrt(temperature)
    # alpha = zeta_cr * psi_prefac / (ngr_ngas * ngr_ngas * k0 * (n_eff / m_neutral))
    # psi_0 = 0.5188025 - 0.804386 * np.log(y)
    # psi = np.repeat(psi_0, len(rho))  # solution for large alpha [>~10]
    # psi[alpha < 0.002] = (alpha * (1.0 - y) / (1.0 + alpha * (1.0 + y)))[alpha < 0.002]
    # psi[alpha < 10.0] = psi_0 / (1.0 + 0.027 / alpha[alpha < 10.0])
    # ##if(alpha<0.002) psi=alpha*(1.-y)/(1.+alpha*(1.+y))else if(alpha<10.) psi=psi_0/(1.+0.027/alpha)# accurate approximation for intermediate values we can use here
    # k_e = k0 * np.exp(psi)  # e-grain collision rate coefficient
    # # i-grain collision rate coefficient
    # k_i = k0 * np.sqrt(ELECTRONMASS_CGS / (m_ion * PROTONMASS_CGS)) * (1 - psi)
    # n_elec = zeta_cr / (ngr_ngas * k_e)  # electron number density
    # n_ion = zeta_cr / (ngr_ngas * k_i)  # ion number density
    # # mean grain charge (note this is signed, will be negative)
    # Z_grain = psi / psi_prefac

    ###### end approximations for when we don't know ionization

    mu_eff = 2.38
    x_elec = (F["PartType0/ElectronAbundance"][:] * HYDROGEN_MASSFRAC * mu_eff).clip(1e-18, 1e100)
    R = x_elec * psi_prefac / ngr_ngas
    # return R
    psi_0 = -3.787124454911839
    # R is essentially the ratio of negative charge in e- to dust: determines which regime we're in to set quantities below
    n_elec = x_elec * n_eff / mu_eff
    #     if(R > 100.) {psi=psi_0;} else if(R < 0.002) {psi=R*(1.-y)/(1.+2.*y*R);} else {psi=psi_0/(1.+pow(R/0.18967,-0.5646));} // simple set of functions to solve for psi, given R above, using the same equations used to determine low-temp ion fractions

    psi = psi_0 / (1.0 + np.power(R / 0.18967, -0.5646))
    psi[R > 100] = psi_0
    psi[R < 0.002] = (R * (1.0 - y) / (1.0 + 2.0 * y * R))[R < 0.002]
    # if(R > 100.) psi=psi_0 else if(R < 0.002) psi=R*(1.-y)/(1.+2.*y*R)else psi=psi_0/(1.+np.power(R/0.18967,-0.5646))# simple set of functions to solve for psi, given R above, using the same equations used to determine low-temp ion fractions
    n_ion = n_elec * y * np.exp(psi) / (1.0 - psi)
    # we can immediately now calculate these from the above
    Z_grain = psi / psi_prefac

    # now define more variables we will need below #
    B_Gauss = (F["PartType0/MagneticField"][:] ** 2).sum(1) ** 0.5 * UNIT_B_IN_GAUSS
    # B_Gauss *= UNIT_B_IN_GAUSS
    xe = n_elec / n_eff
    xi = n_ion / n_eff
    xg = ngr_ngas
    # get collision rates/cross sections for different species #
    # Pinto & Galli 2008
    nu_g = 7.90e-6 * ag01 * ag01 * np.sqrt(temperature / m_neutral) / (m_neutral + m_grain)
    nu_ei = 51.0 * xe * np.power(temperature, -1.5)  # Pandey & Wardle 2008 (e-ion)
    # Pinto & Galli 2008 for latter (e-neutral)
    nu_e = nu_ei + 6.21e-9 * np.power(temperature / 100.0, 0.65) / m_neutral
    # Pandey & Wardle 2008 for former (e-ion)
    nu_ie = ((ELECTRONMASS_CGS * xe) / (m_ion * PROTONMASS_CGS * xi)) * nu_ei
    # Pandey & Wardle 2008 for former (e-ion), Pinto & Galli 2008 for latter (i-neutral)
    nu_i = nu_ie + 1.57e-9 / (m_neutral + m_ion)
    # use the cross sections to determine the hall parameters and conductivities #
    beta_prefac = ELECTRONCHARGE_CGS * B_Gauss / (PROTONMASS_CGS * C_LIGHT_CGS * n_eff)
    # standard beta factors (Hall parameters)
    beta_i = beta_prefac / (m_ion * nu_i)
    beta_e = beta_prefac / (ELECTRONMASS_CGS / PROTONMASS_CGS * nu_e)
    beta_g = beta_prefac / (m_grain * nu_g) * np.abs(Z_grain)
    be_inv = 1 / (1 + beta_e * beta_e)
    bi_inv = 1 / (1 + beta_i * beta_i)
    bg_inv = 1 / (1 + beta_g * beta_g)
    # ohmic conductivity
    sigma_O = xe * beta_e + xi * beta_i + xg * np.abs(Z_grain) * beta_g
    sigma_H = -xe * be_inv + xi * bi_inv + xg * Z_grain * bg_inv  # hall conductivity
    # pedersen conductivity
    sigma_P = xe * beta_e * be_inv + xi * beta_i * bi_inv + xg * np.abs(Z_grain) * beta_g * bg_inv
    sign_Zgrain = Z_grain / np.abs(Z_grain)
    sign_Zgrain[Z_grain == 0] = 0
    # alternative formulation which is automatically positive-definite

    sigma_A2 = (
        (xe * beta_e * be_inv) * (xi * beta_i * bi_inv) * np.power(beta_i + beta_e, 2)
        + (xe * beta_e * be_inv)
        * (xg * np.abs(Z_grain) * beta_g * bg_inv)
        * np.power(sign_Zgrain * beta_g + beta_e, 2)
        + (xi * beta_i * bi_inv)
        * (xg * np.abs(Z_grain) * beta_g * bg_inv)
        * np.power(sign_Zgrain * beta_g - beta_i, 2)
    )
    # now we can finally calculate the diffusivities #
    eta_prefac = B_Gauss * C_LIGHT_CGS / (4 * np.pi * ELECTRONCHARGE_CGS * n_eff)
    eta_O = eta_prefac / sigma_O
    sigma_perp2 = sigma_H * sigma_H + sigma_P * sigma_P
    eta_H = eta_prefac * sigma_H / sigma_perp2
    eta_A = eta_prefac * (sigma_A2) / (sigma_O * sigma_perp2)
    eta_O = np.abs(eta_O)
    # these depend on the absolute values and should be written as such, so eta is always positive [not true for eta_h]
    eta_A = np.abs(eta_A)
    # convert units to code units
    # units_cgs_to_code = UNIT_TIME_IN_CGS / (UNIT_LENGTH_IN_CGS * UNIT_LENGTH_IN_CGS) # convert coefficients (L^2/t) to code units [physical]
    return eta_O, eta_H, eta_A
