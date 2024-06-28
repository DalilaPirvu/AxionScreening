#!/usr/bin/env python3.7.16

# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
print([ii for ii in sys.path])
sys.path.append('/home/dpirvu/axion_new/hmvec-master/')
sys.path.append('/home/dpirvu/python_stuff/')
print([ii for ii in sys.path])
import hmvec as hm
import numpy as np

from compute_power_spectra import *
from params import *
from plotting import *
import random

import functools
from concurrent.futures import ProcessPoolExecutor

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# Select electron profile
conv_gas  = True
conv_NFW  = False
pick_prof = (True if conv_gas else False)

do_basic_cell = True
do_cell_cross_gal = True

multirvir = 1.

# Select DP mass
maind=0

unwise_color = 'blue'
pathdndz = "/home/dpirvu/DarkPhotonxunWISE/dataHOD/normalised_dndz_cosmos_0.txt"
choose_dict = 21

# If parallelized
num_workers = 64

zthr  = 1.9
zreio = 1.9

MA = dictKey[maind]
zMin, zMax, rMin, rMax = chooseModel(MA, modelParams)

if conv_gas:
    name   = 'battagliaAGN'
    rscale = False
elif conv_NFW:
    rscale = True

####### HALO MODEL ########

ellMax    = 9600
ells      = np.arange(ellMax)
chunksize = max(1, len(ells)//num_workers)

#hlil = 0.6766
#mMin = 7e8/hlil
mMin = 1e11
mMax = 1e17

zMax = min(zthr, zMax)
nZs  = 50
nMs  = 100
ms  = np.geomspace(mMin,mMax, nMs)               # masses
zs  = np.linspace(zMin, zMax, nZs)               # redshifts
ks  = np.geomspace(1e-4,1e3, 5001)               # wavenumbers

print('Axion mass:', MA)
print('Halo masses:', mMin, mMax, nMs)
print('Redshifts:', zMin, zMax, nZs)

# Halo Model
hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir', concmode='BHATTACHARYA', unwise_color=unwise_color, choose_dict=choose_dict)
print('Test hmvec')
print(hm.default_params['H0'])
print(hcos.conc)

chis     = hcos.comoving_radial_distance(zs)
rvirs    = hcos.rvir(ms[None,:],zs[:,None])
cs       = hcos.concentration()
Hz       = hcos.h_of_z(zs)
nzm      = hcos.get_nzm()
biases   = hcos.get_bh()
deltav   = hcos.deltav(zs)
rhocritz = hcos.rho_critical_z(zs)

dvols  = get_volume_conv(chis, Hz)
PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)
Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax, PzkLin)
Pzell0 = Pzell.transpose(1,0)
print('Done turning into multipoles.')

hod_name = "unWISE"+unwise_color
hcos.add_hod(name=hod_name)

path_params = np.asarray([nZs, zMin, zMax, ellMax, multirvir, rscale])

print('Importing CMB power spectra and adding temperature monopole.')
CMB_ps        = hcos.CMB_power_spectra()
unlenCMB      = CMB_ps['unlensed_scalar']
unlenCMB      = unlenCMB[:ellMax, :]
unlenCMB[0,0] = TCMB**2.


if do_basic_cell:
    rcross = multirvir*rvirs

    ucosth, angs = get_new_halo_skyprofile(zs, chis, rcross)
    prob = new_conv_prob_gas(zs, rcross)

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))

    # Z_ell has dimensions Nell, Nz, Nm
    probell_temp  = prob[None,...] * uell0
    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_tau = (probell_temp[:ellMax]) * ct[:, None, None]

    avtau, dtaudz = get_avtau(zs, ms, nzm, dvols, zell_tau[0])
    dtaudz_ell    = get_dtauell(ms, nzm, dvols, biases, zell_tau)

    # Assemble power spectra
    int_uell_1h = np.trapz(nzm[None,...] * zell_tau**2.               , ms, axis=-1)
    int_uell_2h = np.trapz(nzm[None,...] * biases[None,...] * zell_tau, ms, axis=-1)

    Cl1h  = np.trapz(dvols[None,:] * int_uell_1h                    , zs, axis=1)
    Cl2h  = np.trapz(dvols[None,:] * np.abs(int_uell_2h)**2. *Pzell0, zs, axis=1)
    Cltot = Cl1h + Cl2h

    scrTT = Cltot * TCMB**2. * units**2.
    np.save(new_test_data(*path_params), [avtau, dtaudz, rcross, uell0, prob, Cl1h, Cl2h, scrTT])
    print('Done basic.')

if do_cell_cross_gal:
    ellMax = 9600
    ells = np.arange(ellMax)

    hcos.add_hod(name=unwise_color)
    Ncs  = hcos.hods[unwise_color]['Nc']
    Nss  = hcos.hods[unwise_color]['Ns']
    ngal = hcos.hods[unwise_color]['ngal']
    hod, uc, us = hcos._get_hod_common(unwise_color)

    dndz, zs, N_gtot, W_g, zsHOD, dndzHOD = get_dndz(zs, pathdndz, dvols)

    uk_g   = (      Ncs[None,:,None] + us     * Nss[None,:,None]    ) / ngal[:,None,None]
    uk_gsq = (2.*us*Nss[None,:,None] + us**2. * Nss[None,:,None]**2.) / ngal[:,None,None]**2.
    PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)

    Pzell, uell_g, uell_gsq = get_fourier_to_multipole_functs(zs, ms, ks, chis, W_g, ellMax, uk_g, uk_gsq, PzkLin)
    print('Done turning into multipoles.')

    avtau, dtaudz, rcross, uell0, prob, Cell1Halo, Cell2Halo, CMBDP = np.load(new_test_data(*path_params))

    screeningProbell = prob[None,...] * uell0
    zell_tau = (screeningProbell[:ellMax]).transpose(1,2,0) * np.sqrt((4.*np.pi)/(2*ells+1))[None,None,:]

    # Assemble power spectra
    int_uell_g_1h = np.trapz(nzm[...,None] * uell_gsq         , ms, axis=1)
    int_zell_taug = np.trapz(nzm[...,None] * zell_tau * uell_g, ms, axis=1)

    int_uell_g_2h = np.trapz(nzm[...,None] * biases[...,None] * uell_g  , ms, axis=1)
    int_zell_tau  = np.trapz(nzm[...,None] * biases[...,None] * zell_tau, ms, axis=1)

    Cell_taug_1h  = np.trapz(dvols[:,None] * int_zell_taug                       , zs, axis=0)
    Cell_taug_2h  = np.trapz(dvols[:,None] * int_zell_tau * int_uell_g_2h * Pzell, zs, axis=0)
    Cell_taug_tot = Cell_taug_1h + Cell_taug_2h

    np.save(new_cl_data_galtau_path(nZs, zMin, zMax, ellMax, name, multirvir, unwise_color, choose_dict), Cell_taug_tot)

    Cell_gg_1h  = np.trapz(dvols[:,None] * int_uell_g_1h                    , zs, axis=0)
    Cell_gg_2h  = np.trapz(dvols[:,None] * np.abs(int_uell_g_2h)**2. * Pzell, zs, axis=0)
    Cell_gg_tot = Cell_gg_1h + Cell_gg_2h

    np.save(new_cl_data_galgal_path(nZs, zMin, zMax, ellMax, name, multirvir, unwise_color, choose_dict), Cell_gg_tot)
    print('Done ', name, unwise_color, choose_dict)
