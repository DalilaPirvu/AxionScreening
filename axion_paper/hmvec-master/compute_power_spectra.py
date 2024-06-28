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
    dndz     = np.interp(zs, zsHOD, dndz_data[1,:])
    N_gtot   = np.trapz(dndz, zs, axis=0)
    W_g      = dndz/N_gtot/dvols
    return dndz, zs, N_gtot, W_g, zsHOD, dndz_data[1,:]

def get_dndzHOD(zs, path, dvols):
    dndz_data= np.transpose(np.loadtxt(path, dtype=float))
    zsHOD    = dndz_data[0,:]
    dndz     = np.interp(zs, zsHOD, dndz_data[1,:])
    return dndz

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


def get_gaussian_pol_1kp(zs, chis, ellMax, Npc=1.):
    # kiloparsecs; Npc = number of kpc
    lendom= 1e-3 * Npc
    rchis = chis*aa(zs)

    sigsq = (lendom / rchis)**2.
    ells  = np.arange(ellMax)[:,None]
    expo  = - ells * (ells + 1.) * sigsq[None,:] / 2.
    axell = (1./15.) * 2.*np.pi*sigsq * np.exp(expo)
    return axell


def get_scrCLs_pol_axion(l0Max, l1Max, l2Max, l1list, PPCl, TCMB):
    ell1_CMB = np.arange(2, l1Max)
    ell2_scr = np.arange(2, l2Max)

    every_pair = np.asarray(list(itertools.product(ell1_CMB, ell2_scr)))
    allcomb = len(every_pair)
    nums = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    scrEE = np.zeros((l0Max))
    for ind, (l1, l2) in enumerate(every_pair):
        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        # index l2 is dark screening (spin 0), index l1 is the polarization (spin 2)
        lo1  = np.argmin(np.abs(l1list - l1))
        norm = PPCl[l2][lo1] * (2.*l1+1.)*(2.*l2+1.)/(4.*np.pi)

        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        mix = norm * np.abs(w220[2][cw:dw+1])**2.
        scrEE[l220] += mix
    return scrEE * TCMB**2.


def get_scrBLs_polaxion_gal(l0Max, l1Max, l2Max, l1list, Blmixx, TCMB):
    ell1_pol  = np.arange(2, l1Max)
    ell2_dscr = np.arange(2, l2Max)

    every_pair = np.asarray(list(itertools.product(ell1_pol, ell2_dscr)))
    allcomb    = len(every_pair)
    nums       = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    scrbispEE  = np.zeros((len(Blmixx), l0Max))
    for ind, (l1,l2) in enumerate(every_pair):
        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        # index l2 is dark screening (spin 0), index l1 is the polarization (spin 2)
        w220   = wigner.wigner_3jj(l1, l2, 2, 0)
        cm, dm = max(2, int(w220[0])), min(int(w220[1]), l0Max-1)
        l220   = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        norm = (2.*l1+1.) * (2.*l2+1.) / (4.*np.pi)
        lo1  = np.argmin(np.abs(l1list - l1))
        Bell = Blmixx[:, l2, lo1]
        mix  = norm * np.abs(w220[2][cw:dw+1])**2.

        scrbispEE[:,l220] += Bell[:,None] * mix[None,:]

    # it returns the linear matter PS ell on position 0 and the other ell on position 1
    return scrbispEE * TCMB**2.

def get_scrBLs_1h_polaxion_gal(lMax, LMax, LLMax, lshortlist, Blmixx, TCMB):
    Lscr1 = np.arange(2, LMax)

    # itertools.combinations_with_replacement doesn't repeat entries
    # eg: between (2,4) and (4,2) only (2,4) is kept
    # so, we multiply each result by 2 unless l1 = l2 
    every_pair = np.asarray(list(itertools.combinations_with_replacement(Lscr1, r=2)))
    allcomb    = len(every_pair)
    nums       = np.array(np.linspace(0, allcomb, 10), dtype=int).tolist()

    scrbispBBg  = np.zeros((lMax, np.shape(Blmixx)[-1]))
    for ind, (l1,l2) in enumerate(every_pair):
        if ind in nums: print(ind, 'out of', allcomb, nums.index(ind)+1)

        w000   = wigner.wigner_3jj(l1, l2, 0, 0)
        cm, dm = max(2, int(w000[0])), min(int(w000[1]), lMax-1)
        l000   = np.arange(cm, dm+1)
        cw, dw = int(cm - w000[0]), int(dm - w000[0])

        multi = 2.
        if l1==l2: multi = 1.
        norm  = (2.*l1+1.) * (2.*l2+1.) / (4.*np.pi)
        mix   = multi * norm * np.abs(w000[2][cw:dw+1])**2.

        lo1  = np.argmin(np.abs(lshortlist - l1))
        lo2  = np.argmin(np.abs(lshortlist - l2))
        Bell = Blmixx[lo1, lo2]

        scrbispBBg[l000,:] += Bell[None,:] * mix[:,None]
    return scrbispBBg * TCMB**2.


def get_bispectrum_constraint(fsky, mm, mmm, Clnoise, Clgg, Blll):
    l1List = np.arange(mm, mmm)
    l2List = np.arange(mm, mmm)
    every_pair = np.asarray(list(itertools.product(l1List, l2List)))

    constraint = 0.
    for ind, (l1, l2) in enumerate(every_pair):
        w220   = wigner.wigner_3jj(l1, l2, 2, -2)
        cm, dm = max(mm, int(w220[0])), min(int(w220[1]), mmm-1)
        l0list = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        times = (2.*l1+1.) * (2.*l2+1.) * (2*l0list+1.) / (4.*np.pi)
        elll  = (1. + (-1.)**(l0list + l1 + l2))
        norm  = elll * times * np.abs(w220[2][cw:dw+1])**2.

        numer = ( 0.5*(Blll[l0list,l1] +  Blll[l0list,l2]) )**2.
        denom = Clnoise[l1] * Clnoise[l2] * Clgg[l0list]
        fact  = norm * numer / denom
        constraint += np.sum(fact)
    return 0.7 / (fsky * constraint)**0.125

def get_1h_bispectrum_constraint(fsky, mm, mmm, Clnoise, Clgg, Blll):
    l1List = np.arange(mm, mmm)
    l2List = np.arange(mm, mmm)
    every_pair = np.asarray(list(itertools.product(l1List, l2List)))

    constraint = 0.
    for ind, (l1, l2) in enumerate(every_pair):
        w220   = wigner.wigner_3jj(l1, l2, 2, -2)
        cm, dm = max(mm, int(w220[0])), min(int(w220[1]), mmm-1)
        l0list = np.arange(cm, dm+1)
        cw, dw = int(cm - w220[0]), int(dm - w220[0])

        times = (2.*l1+1.) * (2.*l2+1.) * (2*l0list+1.) / (4.*np.pi)
        elll  = (1. + (-1.)**(l0list + l1 + l2))
        norm  = elll * times * np.abs(w220[2][cw:dw+1])**2.

        numer = ( 0.5*(Blll[l1,l0list] +  Blll[l2,l0list]) )**2.
        denom = Clnoise[l1] * Clnoise[l2] * Clgg[l0list]
        fact  = norm * numer / denom
        constraint += np.sum(fact)
    return 0.7 / (fsky * constraint)**0.125


def get_TTg_bispectrum_constraint(fsky, mm, mmm, Clnoise, Clgg, Blll):
    l1List = np.arange(mm, mmm)
    l2List = np.arange(mm, mmm)
    every_pair = np.asarray(list(itertools.product(l1List, l2List)))

    constraint = 0.
    for ind, (l1, l2) in enumerate(every_pair):
        w000   = wigner.wigner_3jj(l1, l2, 0, 0)
        cm, dm = max(mm, int(w000[0])), min(int(w000[1]), mmm-1)
        l0list = np.arange(cm, dm+1)
        cw, dw = int(cm - w000[0]), int(dm - w000[0])

        times = (2.*l1+1.) * (2.*l2+1.) * (2*l0list+1.) / (4.*np.pi)
        norm  = times * np.abs(w000[2][cw:dw+1])**2.

        numer = Blll[l1,l2]**2.
        denom = Clnoise[l1] * Clnoise[l2] * Clgg[l0list]
        fact  = norm * numer / denom
        constraint += np.sum(fact)
    return 0.7 / (fsky * constraint)**0.125


def get_noise_and_foregrounds_Planck(TCMB, elllim_Planck, nfreqs_Planck):
    path_dir = '/home/dpirvu/axion/data/foreground_models/'
    pref = TCMB**2. / 0.4

    #Planck_freqs 30, 44, 70, 100, 143, 217, 353, 545, 857
    full_Planck_TT = np.load(path_dir+'dimensionless_Planck_fg_auto_TT_full2D.npy')
    full_Planck_EE = np.load(path_dir+'dimensionless_Planck_fg_auto_EE_full2D.npy')
    full_Planck_BB = np.load(path_dir+'dimensionless_Planck_fg_auto_BB_full2D.npy')

    full_Planck_TT = full_Planck_TT[:elllim_Planck,:nfreqs_Planck,:nfreqs_Planck]
    full_Planck_EE = full_Planck_EE[:elllim_Planck,:nfreqs_Planck,:nfreqs_Planck]
    full_Planck_BB = full_Planck_BB[:elllim_Planck,:nfreqs_Planck,:nfreqs_Planck]
    
    NellTT_Planck, NellEE_Planck, NellBB_Planck, Beams_Planck = noise(elllim_Planck, nfreqs_Planck, Planck)

    Nell_Planck, Fell_Planck = np.zeros((2, 3, elllim_Planck, nfreqs_Planck, nfreqs_Planck))
    Nell_Planck[0,2:] = NellTT_Planck[2:] / Beams_Planck[2:]
    Nell_Planck[1,2:] = NellEE_Planck[2:] / Beams_Planck[2:]
    Nell_Planck[2,2:] = NellBB_Planck[2:] / Beams_Planck[2:]

    Fell_Planck[0,2:] = pref * full_Planck_TT[2:] / Beams_Planck[2:]
    Fell_Planck[1,2:] = pref * full_Planck_EE[2:] / Beams_Planck[2:]
    Fell_Planck[2,2:] = pref * full_Planck_BB[2:] / Beams_Planck[2:]
    return Nell_Planck, Fell_Planck

def get_noise_and_foregrounds_S4(TCMB, elllim_S4, nfreqs_S4):
    path_dir = '/home/dpirvu/axion/data/foreground_models/'
    pref = TCMB**2. / 0.4

    #S4_freqs 20, 27, 39, 93, 145, 225, 278
    full_S4_TT = np.load(path_dir+'full_S4_foregrounds_TT.npy')
    full_S4_EE = np.load(path_dir+'full_S4_foregrounds_EE.npy')
    full_S4_BB = np.load(path_dir+'full_S4_foregrounds_BB.npy')

    full_S4_TT = full_S4_TT[:elllim_S4,:nfreqs_S4,:nfreqs_S4]
    full_S4_EE = full_S4_EE[:elllim_S4,:nfreqs_S4,:nfreqs_S4]
    full_S4_BB = full_S4_BB[:elllim_S4,:nfreqs_S4,:nfreqs_S4]

    NellTT_S4, NellEE_S4, NellBB_S4, Beams_S4 = noise(elllim_S4, nfreqs_S4, CMBS4)

    Nell_S4, Fell_S4 = np.zeros((2, 3, elllim_S4, nfreqs_S4, nfreqs_S4))
    Nell_S4[0,2:] = NellTT_S4[2:] / Beams_S4[2:]
    Nell_S4[1,2:] = NellEE_S4[2:] / Beams_S4[2:]
    Nell_S4[2,2:] = NellBB_S4[2:] / Beams_S4[2:]

    Fell_S4[0,2:] = (NellTT_S4[2:] + pref * full_S4_TT[2:]) / Beams_S4[2:]
    Fell_S4[1,2:] = (NellEE_S4[2:] + pref * full_S4_EE[2:]) / Beams_S4[2:]
    Fell_S4[2,2:] = (NellBB_S4[2:] + pref * full_S4_BB[2:]) / Beams_S4[2:]
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
        NellTT[2:,frq,frq]= deltaT[frq] * ( 1. + rednoise )
        Beams[2:,frq] = np.exp(-ellexpo * beamFWHM[frq])

    Beams2D  = (Beams[:,None,:] * Beams[:,:,None])**0.5
    return NellTT, np.sqrt(2)*NellTT, np.sqrt(2)*NellTT, Beams2D

def get_ILC_noise(ellMax, units0, screening, foregs, recCMB, experiment, nspec=3, nfreqs=7):
    ells   = np.arange(ellMax)
    freqs  = experiment['freqseV'][:nfreqs]
    units  = xov(freqs) * freqs
    freqMAT= units0**2. / np.outer(units, units)
    ee     = np.ones(nfreqs)
    onesMAT= np.outer(ee, ee)

    weights = np.zeros((nspec, ellMax, nfreqs))
    leftover= np.zeros((nspec, ellMax))
    elltodo = np.arange(2, ellMax)
    for spec in range(nspec):
        for ell in elltodo:
            CellBBω2= freqMAT * recCMB[spec,ell]
            Cellττ  = onesMAT * screening[spec,ell]
            Fellω2  = freqMAT * foregs[spec,ell]
            Cellinv = scp.linalg.inv(CellBBω2 + Fellω2 - Cellττ)

            weights[spec,ell] = (Cellinv@ee)/(ee@Cellinv@ee)
       #     if spec==0 and ell%100==10:
       #         print(spec, ell, weights[spec,ell], np.sum(weights[spec,ell]))
            leftover[spec,ell]= weights[spec,ell]@(CellBBω2 + Fellω2)@weights[spec,ell]
    return weights, leftover




def sigma_screening(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[0, :ellmax] + leftover[0, :ellmax]
    ClEENl = epsilon4 * screening[1, :ellmax] + leftover[1, :ellmax]
    ClBBNl = epsilon4 * screening[2, :ellmax] + leftover[2, :ellmax]

    dClTTde4 = screening[0, :ellmax]
    dClEEde4 = screening[1, :ellmax]
    dClBBde4 = screening[2, :ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCov    = np.diag([ClTTNl[el], ClEENl[el], ClBBNl[el]])
        CCovInv = np.linalg.inv(CCov)
        dCovde4 = np.diag([dClTTde4[el], dClEEde4[el], dClBBde4[el]])
        TrF[el] = 0.5*(2.*el+1.)*np.trace(CCovInv@dCovde4@CCovInv@dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screening_TT(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[0, :ellmax] + leftover[0, :ellmax]
    ClEENl = epsilon4 * screening[1, :ellmax] + leftover[1, :ellmax]
    ClBBNl = epsilon4 * screening[2, :ellmax] + leftover[2, :ellmax]

    dClTTde4 = screening[0, :ellmax]
    dClEEde4 = screening[1, :ellmax]
    dClBBde4 = screening[2, :ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCovInv = 1./ClTTNl[el]
        dCovde4 = dClTTde4[el]
        TrF[el] = 0.5*(2.*el+1.) * (CCovInv*dCovde4*CCovInv*dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screening_BB(epsilon4, fsky, ellmin, ellmax, screening, leftover):
    #print('Full ', np.shape(leftover), np.shape(screening))
    ClTTNl = epsilon4 * screening[0, :ellmax] + leftover[0, :ellmax]
    ClEENl = epsilon4 * screening[1, :ellmax] + leftover[1, :ellmax]
    ClBBNl = epsilon4 * screening[2, :ellmax] + leftover[2, :ellmax]

    dClTTde4 = screening[0, :ellmax]
    dClEEde4 = screening[1, :ellmax]
    dClBBde4 = screening[2, :ellmax]

    TrF = np.zeros(ellmax)
    for el in range(ellmin, ellmax):
        CCovInv = 1./ClBBNl[el]
        dCovde4 = dClBBde4[el]
        TrF[el] = 0.5*(2.*el+1.) * (CCovInv*dCovde4*CCovInv*dCovde4)
    return 0.7 / (fsky * np.sum(TrF))**0.125

def sigma_screeningVunWISE(ep2, fsky, ellmin, ellmax, cltauscreening, leftover, clgaltau, clgalgal):
    ClTTNl      = ep2**2.* cltauscreening[0,:ellmax] + leftover[0,:ellmax]
    Clττ        = clgalgal
    ClTτscr     = ep2    * clgaltau
    dClTTde2    = 2.*ep2 * cltauscreening[0,:ellmax]
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
    ClTTNl      = ep2**2.* cltauscreening[0,:ellmax] + leftover[0,:ellmax]
    Clττ        = templategal[:ellmax]
    ClTτscr     = ep2    * templategal[:ellmax] * TCMB#/np.sqrt(4.*np.pi)
    dClTTde2    = 2.*ep2 * cltauscreening[0,:ellmax]
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


############### COMPUTE ANGULAR POWER SPECTRA ###########################

def get_rcross_per_halo(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, pick_prof, name='battagliaAGN'):
    if pick_prof:
        return get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, name)
    else:
        print('Error: NFW profile not implemented')
        return None

def get_rcross_per_halo_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, name='battagliaAGN'):
    """ Compute crossing radius of each halo
    i.e. radius where plasma mass^2 = dark photon mass^2
    Find the index of the radius array where plasmon mass^2 = dark photon mass^2 """

    m200critz, r200critz = get_200critz(zs, ms, cs, rhocritz, deltav)

    rcross_res = np.zeros((len(zs), len(ms)))
    for i_z, z in enumerate(zs):
        for i_m, m in enumerate(ms):
            func = lambda x: np.abs(get_gas_profile(x, z, m200critz[i_z, i_m], r200critz[i_z, i_m], rhocritz[i_z], name=name) * conv/MA**2. - 1.)
            rcross_res[i_z, i_m] = fsolve(func, x0=rs[0])
    return rcross_res


def conv_prob(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=False, name='battagliaAGN'):
    if pick_prof:
        return conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, name)
    else:
        print('Error: NFW profile not implemented')
        return None

def conv_prob_flat(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=False, name='battagliaAGN'):
    if pick_prof:
        return conv_prob_flat_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, name)
    else:
        print('Error: NFW profile not implemented')
        return None

def prob_theta(zs, ms, rs, rvir, rhocritz, deltav, cs, mDP, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=False, name='battagliaAGN'):
    return 2.*np.heaviside(rvir - rcross, 0.5)

def conv_prob_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, name='battagliaAGN'):
    """Conversion probability including the B field profile"""
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    drhodr = get_deriv_gas_profile(rcross, zs[:,None], m200, r200, rhocritz[:,None], name=name)
    dmdr   = np.abs(conv*drhodr)
    omgz   = (1.+zs[:,None])# * omega0 but we want to keep frequency dependence separate

    bprof = get_B_rcross(zs, ms, m200, r200, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins)
    units = np.pi/3. * gauss2evsq(1.)**2. * mpcEVinv
    return units * bprof**2. * omgz / dmdr

def conv_prob_flat_gas(zs, ms, rs, rvir, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, name='battagliaAGN'):
    """Conversion probability including the B field profile"""
    m200, r200 = get_200critz(zs, ms, cs, rhocritz, deltav)
    drhodr = get_deriv_gas_profile(rcross, zs[:,None], m200, r200, rhocritz[:,None], name=name)
    dmdr   = np.abs(conv*drhodr)
    omgz   = (1.+zs[:,None])# * omega0 but we want to keep frequency dependence separate

    bprof = get_B_rcross_flat(zs, ms, m200, r200, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins)
    units = np.pi/3. * gauss2evsq(1.)**2. * mpcEVinv
    return units * bprof**2. * omgz / dmdr


def get_B_rcross(zs, ms, m200c, r200c, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins):
    rcross_ratio = rcross/r200c
    Brcross = np.zeros((len(zs), len(ms)))

    ms_ind = np.digitize(m200c[0, :], mass_bins)
    ms_ind[ms_ind == len(logB_interp_list)] = len(logB_interp_list)-1.

    for i_m in range(len(ms)):
        for i_z, z_val in enumerate(zs):        
            if rcross_ratio[i_z, i_m] < rad_bins[0]:
                Brcross[i_z, i_m] = ( get_pth_profile(rcross_ratio[i_z, i_m]*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) /
                                      get_pth_profile(rad_bins[0]*r200c[i_z, i_m], z_val, m200c[i_z, i_m], r200c[i_z, i_m], rhocritz[i_z]) ) * 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
            else:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rcross_ratio[i_z, i_m] ] )     
    return Brcross

def get_B_rcross_flat(zs, ms, m200c, r200c, rhocritz, rcross, logB_interp_list, mass_bins, rad_bins):
    rcross_ratio = rcross/r200c
    Brcross = np.zeros((len(zs), len(ms)))

    ms_ind = np.digitize(m200c[0, :], mass_bins)
    ms_ind[ms_ind == len(logB_interp_list)] = len(logB_interp_list)-1

    for i_m in range(len(ms)):
        for i_z, z_val in enumerate(zs):        
            if rcross_ratio[i_z, i_m] < rad_bins[0]:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rad_bins[0]] )   
            else:
                Brcross[i_z, i_m] = 10.**logB_interp_list[ms_ind[i_m]]( [np.log10(z_val), rcross_ratio[i_z, i_m] ] )     
    return Brcross


def get_volume_conv(chis, Hz):
    # Volume of redshift bin divided by Hubble volume
    # Chain rule factor when converting from integral over chi to integral over z
    return chis**2. / Hz

def get_200critz(zs, ms, cs, rhocritz, deltav):
    delta_rhos1 = deltav*rhocritz
    delta_rhos2 = 200.*rhocritz
    m200critz = hm.mdelta_from_mdelta(ms, cs, delta_rhos1, delta_rhos2)
    r200critz = hm.R_from_M(m200critz, rhocritz[:,None], delta=200.)
    return m200critz, r200critz

def get_halo_skyprofile(zs, chis, rcross):
    # get bounds of each regime within halo
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])[None,...]

    listincr = 1. - np.geomspace(1e-3, 1., 41)
    listincr = np.asarray([1.] + listincr.tolist())[::-1]
    angs = listincr[:,None,None] * fract

    ucosth = (1.-(angs/fract)**2.)**(-0.5)
    ucosth[angs == fract] = 0.
    return ucosth, angs


def get_u00(zs, chis, rcross):
    # this gives the analytical result for the monopole
    rchis = chis*aa(zs)
    fract = (rcross/rchis[:,None])
    return fract**2./2.

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

def get_Celldtaudtau_1h(zs, ms, ks, nzm, dvol, probell, ellMax):
    # The 1-halo term
    ells = np.arange(ellMax)
    Cl1hdz = np.trapz(nzm[None,...] * np.abs(probell)**2., ms, axis=2)
    Cl1h = np.trapz(Cl1hdz * dvol[None,:], zs, axis=1)
    return Cl1h * (4.*np.pi) / (2.*ells+1.)

def get_dCelldtaudtaudz_1h(zs, ms, ks, nzm, dvol, probell, utheta, ellMax):
    # The 1-halo term but do not integrate over redshift
    Cl1hdz = np.trapz(nzm[None,...] * np.abs(utheta[None,...] * np.abs(probell)**2.), ms, axis=2)
    return Cl1hdz * dvol[None,:]


def get_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    if name!='battagliaAGN':
        print('This gas profile is not implemented. Using battagliaAGN instead.')
    rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(beta+gamma)/alpha # gamma sign must be opposite from Battaglia/ACT paper; typo
    return rho * (x**gamma) * ((1.+x**alpha)**expo)

def get_deriv_gas_profile(rs, zs, m200, r200, rhocritz, name='battagliaAGN'):
    if name!='battagliaAGN':
        print('This gas profile is not implemented. Using battagliaAGN instead.')
    rho0, alpha, beta, gamma, xc = battagliaAGN(m200, zs)

    rho = rhocritz * rho0
    x = rs/r200/xc
    expo = -(alpha+beta+gamma)/alpha
    
    drhodr = rho * (x**gamma) * (1. + x**alpha)**expo * (gamma - x**alpha * beta) / rs

    if hasattr(rs, "__len__"): drhodr[rs==0.] = 0.
    elif rs==0: drhodr = 0.
    return drhodr



def w000(Ell, ell0, ell1, ell2):
    # fast wigner 3j with m1 = m2 = m3 = 0
    g = Ell/2.
    w = np.exp(0.5*(lgamma(2.*g-2.*ell0+1.)+lgamma(2.*g-2.*ell1+1.)+lgamma(2.*g-2.*ell2+1.)-lgamma(2.*g+2.))+lgamma(g+1.)-lgamma(g-ell0+1.)-lgamma(g-ell1+1.)-lgamma(g-ell2+1.))
    return w * (-1.)**g


def battagliaAGN(m200, zs):
    # power law fits:
    rho0  = 4000. * (m200/1e14)**0.29    * (1.+zs)**(-0.66)
    alpha = 0.88  * (m200/1e14)**(-0.03) * (1.+zs)**0.19
    beta  = 3.83  * (m200/1e14)**0.04    * (1.+zs)**(-0.025)
        
    gamma = -0.2
    xc    = 0.5
    return rho0, alpha, beta, gamma, xc

def limber_int(ells,zs,ks,Pzks,hzs,chis):
    hzs = np.array(hzs).reshape(-1)
    chis = np.array(chis).reshape(-1)
    prefactor = hzs / chis**2.

    f = interp2d(ks, zs, Pzks, bounds_error=True)     

    Cells = np.zeros(ells.shape)
    for ii, ell in enumerate(ells):
        kevals = (ell+0.5)/chis

        # hack suggested in https://stackoverflow.com/questions/47087109/evaluate-the-output-from-scipy-2d-interpolation-along-a-curve
        # to get around scipy.interpolate limitations
        interpolated = si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], kevals, zs)[0]

        Cells[ii] = np.trapz(interpolated*prefactor, zs)
    return Cells

def get_pth_profile(rs, zs, m200, r200, rhocritz):  # Eq. (17) from ACT https://arxiv.org/pdf/2009.05558.pdf
    xct = 0.497 * (m200/1e14)**(-0.00865) * (1.+zs)**0.731
    x = rs/r200
    gammat = -0.3
    P0 = 2
    alphat = 0.8
    betat = 2.6
    fb = 0.044/0.25
    
    PGNFW = P0 * (x/xct)**gammat * ( 1 + (x/xct)**alphat )**(-betat)
    P200 = m200*200*rhocritz*fb/(2*r200)    # this has an additional factor of G in front, but we don't care about the units since this is only for modeling the B field profile
    
    return PGNFW*P200
