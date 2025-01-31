import h5py
import emcee
from numba import vectorize, cfunc, njit
import numpy as np


@vectorize
def massfunc_kroupa(m, log_m12, log_m23, alpha1, alpha2, alpha3):
    """Kroupa mass function form, with 3-piece power law with slopes alpha1,alpha2, and alpha3 broken at m12 and m23"""
    # if m<mmin or m>mmax:
    #        return 0.
    logm = np.log10(m)
    val = 1.0
    # if logm < log_m12:
    val *= m**alpha1
    if logm > log_m12:
        val *= (m / 10**log_m12) ** (alpha2 - alpha1)
    if logm > log_m23:
        val *= (m / 10**log_m23) ** (alpha3 - alpha2)
    return val


@vectorize
def massfunc_parravano(m, log_mch, gamma, alpha):
    return m**alpha * (1 - np.exp(-((m / 10**log_mch) ** (gamma + (-alpha - 1)))))


@vectorize
def massfunc_chabrier(m, log_mpeak, sigma, alpha):
    mbreak = 1.0
    mpeak = 10**log_mpeak
    if m > mbreak:
        massfunc = 1 / mbreak * np.exp(-np.log10(mbreak / mpeak) ** 2 / (2 * sigma**2)) * (m / mbreak) ** alpha
    else:
        massfunc = 1 / m * np.exp(-np.log10(m / mpeak) ** 2 / (2 * sigma**2))
    return massfunc


def FitMassFunction(
    masses,
    mmin=None,
    mmax=None,
    form="chabrier",
    walkers=100,
    chainlength=1000,
    burnin=100,
    return_samples=False,
):
    """MCMC IMF fitting routine. Returns either the median and +/- sigma quantiles of the likelihood distribution of IMF parameters, or the entire sample from the MCMC run if return_samples==True"""
    masses = np.copy(masses)  # don't overwrite
    if mmax is None:
        mmax = masses.max()
    if mmin is None:
        mmin = masses.min()

    masses = masses[(masses >= mmin) * (masses <= mmax)]
    mgrid = 10 ** np.linspace(
        np.log10(mmin), np.log10(mmax), 1000
    )  # set up a grid to perform the numerical quadrature of the mass function for normalization

    if form == "parravano":
        func = massfunc_parravano
        ndim, nwalkers = 3, walkers
        p0 = np.array([0.5, 0.5, -2])
        p0 = p0 + np.random.normal(size=(nwalkers, ndim)) * 0.1

        @njit
        def log_prob(x, mstar):
            mch, gamma, alpha = x
            massfunc_norm = np.trapz(func(mgrid, mch, gamma, alpha), mgrid)
            massfunc_vals = func(mstar, mch, gamma, alpha) / massfunc_norm
            if np.any(np.isnan(np.log(massfunc_vals))) or np.any(np.isinf(np.log(massfunc_vals))):
                return -np.inf  # 0 likelihood for erroneous values
            return np.sum(
                np.log(massfunc_vals)
            )  # return total log-likelihood of IMF values evaluated at the data points

    elif form == "chabrier":
        func = massfunc_chabrier
        ndim, nwalkers = 3, walkers
        p0 = np.array([-0.5, 0.5, -2])
        p0 = p0 + np.random.normal(size=(nwalkers, ndim)) * 0.1

        @njit
        def log_prob(x, mstar):
            mpeak, sigma, alpha = x
            if sigma < 0.1:
                return -np.inf
            if sigma > 3:
                return -np.inf
            if mpeak < -2:
                return -np.inf
            if mpeak > 1:
                return -np.inf
            f = func(mgrid, mpeak, sigma, alpha)
            #            print(f)
            #            massfunc_norm = np.trapz(f, mgrid)
            #            mtot = np.trapz(f*mgrid,mgrid)
            #            if mtot == 0: print(mpeak,sigma,alpha, f) #return -np.inf
            #            massfunc_norm = mstar.sum() / mtot
            massfunc_norm = 1 / np.trapz(f, mgrid)
            #            print(np.trapz(f*massfunc_norm * mgrid, mgrid))
            massfunc_vals = func(mstar, mpeak, sigma, alpha) * massfunc_norm
            if np.any(np.isnan(np.log(massfunc_vals))) or np.any(np.isinf(np.log(massfunc_vals))):
                return -np.inf  # 0 likelihood for erroneous values
            return np.sum(
                np.log(massfunc_vals)
            )  # return total log-likelihood of IMF values evaluated at the data points

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[masses])
    sampler.run_mcmc(p0, chainlength, progress=True)

    samples = sampler.get_chain(discard=burnin, flat=True)  # throw out the first 1000 as burnin
    if return_samples:
        return samples
    else:
        return np.percentile(samples, [16, 50, 84], axis=0)
