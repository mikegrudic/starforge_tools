import h5py
import emcee
from numba import vectorize, cfunc, njit
import numpy as np
    
@vectorize(fastmath=True)
def massfunc_kroupa(m, log_m12, log_m23, alpha1,alpha2,alpha3):    
    """Kroupa mass function form, with 3-piece power law with slopes alpha1,alpha2, and alpha3 broken at m12 and m23"""
    #if m<mmin or m>mmax:
#        return 0.
    logm = np.log10(m)    
    val = 1.
    #if logm < log_m12:
    val *= m**alpha1
    if logm > log_m12:
        val *= (m/10**log_m12)**(alpha2-alpha1)
    if logm > log_m23:
        val *= (m/10**log_m23)**(alpha3-alpha2)
    return val

def FitMassFunction(masses,mmin=None,mmax=None,func=massfunc_kroupa, walkers=100,chainlength=1000,burnin=100, return_samples=False):
    """MCMC IMF fitting routine. Returns either the median and +/- sigma quantiles of the likelihood distribution of IMF parameters, or the entire sample from the MCMC run if return_samples==True"""
    masses = np.copy(masses) # don't overwrite
    if mmax is None: mmax = masses.max()
    if mmin is None: mmin = masses.min()
        
    masses = masses[(masses>=mmin)*(masses <= mmax)]
    mgrid = 10**np.linspace(np.log10(mmin),np.log10(mmax),1000) # set up a grid to perform the numerical quadrature of the mass function for normalization
    
    @njit(fastmath=True)
    def log_prob(x, mstar):
        log_m12, log_m23, alpha1, alpha2, alpha3 = x # unpack the parameters
        if log_m12 > log_m23: return -np.inf # enforce m12 < m23
        massfunc_norm = np.trapz(func(mgrid, log_m12, log_m23, alpha1, alpha2, alpha3), mgrid) # compute the integral of the un-normalized mass function
        massfunc_vals = func(mstar, log_m12, log_m23,alpha1, alpha2, alpha3)/massfunc_norm # normalize the mass function sampled at the data points
        if np.any(np.isnan(np.log(massfunc_vals))) or np.any(np.isinf(np.log(massfunc_vals))): return -np.inf # 0 likelihood for erroneous values
        return np.sum(np.log(massfunc_vals)) # return total log-likelihood of IMF values evaluated at the data points
        
    ndim, nwalkers = 5, walkers
    p0 = np.array([0,0,0,-1,-2])
    p0 = p0 + np.random.normal(size=(nwalkers,ndim))*0.1
    

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[masses])
    sampler.run_mcmc(p0, chainlength,progress=True)
    
    samples = sampler.get_chain(discard=burnin,flat=True) # throw out the first 1000 as burnin
    if return_samples:
        return samples
    else: 
        return np.percentile(samples,[16,50,84],axis=0)
    
