import hmvec as hm
import numpy as np
import scipy as scp
from scipy.special import eval_legendre, legendre, spherical_jn
import itertools
import wigner
from sympy.physics.wigner import wigner_3j
import time
from scipy import interpolate
from itertools import cycle
from math import atan2,degrees,lgamma 
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp2d,interp1d
import scipy.interpolate as si

from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator
import random
import seaborn as sns

from params import *
#from plotting import *

############### Compute Polarization Screening ###########################

def get_dndz(zs, path, dvols):
    dndz_data= np.transpose(np.loadtxt(path, dtype=float))
    zsHOD    = dndz_data[0,:]
    dndz     = np.interp(zs, dndz_data[0,:], dndz_data[1,:])
    N_gtot   = np.trapz(dndz, zs, axis=0)
    W_g      = dndz/N_gtot/dvols
    return dndz, zs, N_gtot, W_g, dndz_data[0,:], dndz_data[1,:]

def get_fourier_to_multipole_functs(zs, ms, ks, chis, W_g, ellMax, uk_g, uk_gsq, Pklin):
    ells = np.arange(ellMax)
    uell_profile, uellsq_profile = np.zeros((2, len(zs), len(ms), len(ells)))
    Pzell = np.zeros((len(zs), len(ells)))

    f = interp2d(ks, zs, Pklin, bounds_error=True)     
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]
        Pzell[:, ii] = interpolated

    for mi, mm in enumerate(ms):
        f1 = interp2d(ks, zs, uk_g[:,mi,:], bounds_error=True)     
        for ii, ell in enumerate(ells):
            kevals = (ell+0.5)/chis
            interpolated = si.dfitpack.bispeu(f1.tck[0], f1.tck[1], f1.tck[2], f1.tck[3], f1.tck[4], kevals, zs)[0]
            uell_profile[:, mi, ii] = interpolated

        f3 = interp2d(ks, zs, uk_gsq[:,mi,:], bounds_error=True)     
        for ii, ell in enumerate(ells):
            kevals = (ell+0.5)/chis
            interpolated = si.dfitpack.bispeu(f3.tck[0], f3.tck[1], f3.tck[2], f3.tck[3], f3.tck[4], kevals, zs)[0]
            uellsq_profile[:, mi, ii] = interpolated

    uell_g   =    W_g   [:,None,None] * uell_profile
    uell_gsq = (W_g**2.)[:,None,None] * uellsq_profile
    return Pzell, uell_g, uell_gsq

def get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax, Pklin):
    ells = np.arange(ellMax)
    Pzell = np.zeros((len(zs), len(ells)))

    f = interp2d(ks, zs, Pklin, bounds_error=True)     
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]
        Pzell[:, ii] = interpolated
    return Pzell

def get_noise_and_foregrounds_Planck(TCMB, elllim_Planck, nfreqs_Planck):
    path_dir = '/home/dpirvu/axion/data/foreground_models/'
    pref = TCMB**2. / 0.4

    #Planck_freqs 30, 44, 70, 100, 143, 217, 353, 545, 857
    full_Planck_TT = np.load(path_dir+'dimensionless_Planck_fg_auto_TT_full2D.npy')

    full_Planck_TT = full_Planck_TT[:elllim_Planck,:nfreqs_Planck,:nfreqs_Planck]
    
    NellTT_Planck, _, _, Beams_Planck = noise(elllim_Planck, nfreqs_Planck, Planck)

    Nell_Planck, Fell_Planck = np.zeros((2, elllim_Planck, nfreqs_Planck, nfreqs_Planck))
    Nell_Planck[2:] = NellTT_Planck[2:] / Beams_Planck[2:]
    Fell_Planck[2:] = pref * full_Planck_TT[2:] / Beams_Planck[2:]
    return Nell_Planck, Fell_Planck

def get_noise_and_foregrounds_S4(TCMB, elllim_S4, nfreqs_S4):
    path_dir = '/home/dpirvu/axion/data/foreground_models/'
    pref = TCMB**2. / 0.4

    #S4_freqs 20, 27, 39, 93, 145, 225, 278
    full_S4_TT = np.load(path_dir+'full_S4_foregrounds_TT.npy')

    full_S4_TT = full_S4_TT[:elllim_S4,:nfreqs_S4,:nfreqs_S4]

    NellTT_S4, _, _, Beams_S4 = noise(elllim_S4, nfreqs_S4, CMBS4)

    Nell_S4, Fell_S4 = np.zeros((2, elllim_S4, nfreqs_S4, nfreqs_S4))
    Nell_S4[2:] = NellTT_S4[2:] / Beams_S4[2:]

    Fell_S4[2:] = (NellTT_S4[2:] + pref * full_S4_TT[2:]) / Beams_S4[2:]
    return Nell_S4, Fell_S4

def noise(ellMax, nfreqs, experiment):
    # Instrumental noise: takes parameters Beam FWHM and Experiment sensitivity in T
    ''' Output format: (spectrum type, ells, channels)'''

    beamFWHM = experiment['FWHMrad']**2.
    beamFWHM = beamFWHM[:nfreqs]
    deltaT   = experiment['SensitivityμK']**2.
    deltaT   = deltaT[:nfreqs]
    lknee    = experiment['Knee ell']
    aknee    = experiment['Exponent']

    ells = np.arange(2, ellMax)
    rednoise = ((ells/lknee)**aknee if (lknee!=0. and aknee!=0.) else 0.)
    ellexpo  = ells * (ells + 1.) / (8. * np.log(2))

    NellTT = np.zeros((ellMax, nfreqs, nfreqs))
    Beams  = np.zeros((ellMax, nfreqs))
    for frq in range(nfreqs):
        NellTT[2:,frq,frq]= deltaT[frq] #* ( 1. + rednoise )
        Beams[2:,frq] = np.exp(-ellexpo * beamFWHM[frq])

    Beams2D  = (Beams[:,None,:] * Beams[:,:,None])**0.5
    return NellTT, np.sqrt(2)*NellTT, np.sqrt(2)*NellTT, Beams2D

def get_ILC_noise(ellMax, units0, screening, foregs, recCMB, experiment, nspec=1, nfreqs=7):
    ells   = np.arange(ellMax)
    freqs  = experiment['freqseV'][:nfreqs]
    units  = xov(freqs) * freqs**2.
    freqMAT= units0**2. / np.outer(units, units)
    ee     = np.ones(nfreqs)
    onesMAT= np.outer(ee, ee)

    weights = np.zeros((ellMax, nfreqs))
    leftover= np.zeros((ellMax))
    elltodo = np.arange(2, ellMax)
    for ell in elltodo:
        CellBBω2= freqMAT * recCMB[ell]
        Cellττ  = onesMAT * screening[ell]
        Fellω2  = freqMAT * foregs[ell]
        Cellinv = scp.linalg.inv(CellBBω2 + Fellω2 - Cellττ)

        weights[ell] = (Cellinv@ee)/(ee@Cellinv@ee)
        leftover[ell]= weights[ell]@(CellBBω2 + Fellω2)@weights[ell]
    return weights, leftover

def sigma_screening_TT(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[:ellmax] + leftover[:ellmax]

    dClTTde4 = screening[:ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCovInv = 1./ClTTNl[el]
        dCovde4 = dClTTde4[el]
        TrF[el] = 0.5*(2.*el+1.) * (CCovInv*dCovde4*CCovInv*dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screeningVunWISE(ep2, fsky, ellmin, ellmax, cltauscreening, leftover, clgaltau, clgalgal):
    ClTTNl      = ep2**2.* cltauscreening[:ellmax] + leftover[:ellmax]
    Clττ        = clgalgal
    ClTτscr     = ep2    * clgaltau
    dClTTde2    = 2.*ep2 * cltauscreening[:ellmax]
    dClTτscrde2 = clgaltau

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.76 / (fsky * np.sum(TrF))**0.25

def sigma_screeningVtemplate(TCMB, ep2, fsky, ellmin, ellmax, cltauscreening, leftover, templategal):
    ClTTNl      = ep2**2.* cltauscreening[:ellmax] + leftover[:ellmax]
    Clττ        = templategal[:ellmax]
    ClTτscr     = ep2    * templategal[:ellmax] * TCMB#/np.sqrt(4.*np.pi)
    dClTTde2    = 2.*ep2 * cltauscreening[:ellmax]
    dClTτscrde2 = templategal[:ellmax] * TCMB#/np.sqrt(4.*np.pi)

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov = np.asarray([[Clττ[el]   , ClTτscr[el]],\
                           [ClTτscr[el], ClTTNl[el] ]])
        CCovInv = np.linalg.inv(CCov)
        dCovde2 = np.asarray([[0.             , dClTτscrde2[el]],\
                              [dClTτscrde2[el], dClTTde2[el]   ]])

        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde2@CCovInv@dCovde2)
    return 0.76 / (fsky * np.sum(TrF))**0.25


def new_conv_prob_gas(zs, rcross):
    """Conversion probability including the B field profile"""
    return gauss2evsq(1.)**2. * (1.+zs[:,None])**2. * rcross

def get_volume_conv(chis, Hz):
    # Volume of redshift bin divided by Hubble volume
    # Chain rule factor when converting from integral over chi to integral over z
    return chis**2. / Hz

def get_new_halo_skyprofile(zs, chis, rcross):
    # get bounds of each regime within halo
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])[None,...]

    listincr = 1. - np.geomspace(1e-3, 1., 50)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = listincr[:,None,None] * fract

    ucosth = (1.-(angs/fract)**2.)**0.5
    ucosth[angs == fract] = 0.
    return ucosth, angs

def get_uell0(angs, ucosth, ell):
    # this returns the analytical approximation for low ell
    # or numerical result for higher multipoles
 
    uL0 = np.zeros(angs[0].shape)

    approx = ell < 0.1/angs[-1, :, :] # indices for which we can use the approximation

    uL0[approx] = 2.*np.pi * (angs[-1, :, :][approx])**2.

    # angular function u(theta) is projected into multipoles
    cos_angs = np.cos(angs[:, ~approx])
    Pell     = eval_legendre(ell, cos_angs)
    integr   = Pell * np.sin(angs[:, ~approx]) * ucosth[:, ~approx]
    uL0[~approx] = 2.*np.pi * np.trapz(integr, angs[:,~approx], axis=0)

    if ell%100==0: print(ell)
    return uL0 * ((4.*np.pi) / (2.*ell+1.))**(-0.5)

def get_avtau(zs, ms, nzm, dvol, prob00):
    # Average optical depth per redshift bin
    dtaudz = np.trapz(nzm * prob00, ms, axis=-1) * dvol * 4*np.pi
    avtau  = np.trapz(dtaudz, zs, axis=0)
    return avtau, dtaudz

def get_dtauell(ms, nzm, dvol, biases, probell):
    # integrand in 2-halo numerator
    return np.trapz(biases[None,...]*nzm[None,...]*probell, ms, axis=2) * dvol[None,:]
