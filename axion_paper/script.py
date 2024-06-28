#!/usr/bin/env python3.7.16

# To run this script, in a separate terminal type:
#### conda activate conda_env
#### python3 script.py >> ./data/output.txt

import os,sys
print([ii for ii in sys.path])
#sys.path.remove('/home/dpirvu/DarkPhotonxunWISE/hmvec-master')
sys.path.append('/home/dpirvu/axion/hmvec-master/')
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

# Select DP mass
maind=0

unWISEcol = 'blue'
pathdndz = "/home/dpirvu/DarkPhotonxunWISE/dataHOD/normalised_dndz_cosmos_0.txt"

# If parallelized
num_workers = 40

zthr  = 1.9
zreio = 1.9

MA = dictKey[maind]
zMin, zMax, rMin, rMax = chooseModel(MA, modelParams)

if conv_gas:
    name   = 'battagliaAGN'
    rscale = False
elif conv_NFW:
    rscale = True

# B-field stuff from Cristina
# See Table 1 of 2309.13104 for the halo properties for the given mass bins
file_names = ['profile_bfld_halo_1e10_h12.txt', 'profile_bfld_halo_1e10_h11.txt', 'profile_bfld_halo_1e11_h10.txt', 
              'profile_bfld_halo_1e11_h4.txt', 'profile_bfld_halo_1e12_h12.txt', 'profile_bfld_halo_1e13_h4.txt', 
              'profile_bfld_halo_1e13_h8.txt']
# file_names = ['profile_bfld_halo_'+str(i+1)+'.txt' for i in range(7)]# os.listdir('./data/bfield_profiles/')
mass_bins = 10.**np.array([9.9, 10.4, 10.9, 11.4, 12, 12.5, 13])

# Radial bins are the same for all of the files
rad_bins   = np.genfromtxt('./data/profiles/'+file_names[0], skip_header=3, max_rows=1)
rad_bins_c = rad_bins[:-1]+(rad_bins[1:]-rad_bins[:-1])/2.

Bfiled_grid = np.zeros((len(mass_bins), 66, 23))
logB_interp_list = []

for i, file in enumerate(file_names):
    # in gauss
    Bfiled_grid[i] = np.genfromtxt('./data/profiles/'+file_names[i], skip_header=7).astype(float)

#  logB_interp_list.append(RegularGridInterpolator((np.log10(Bfiled_grid[i][::, 0]), rad_bins_c), \
#                                                   np.log10(Bfiled_grid[i][::, 3:]), \
#                                                   bounds_error=False, fill_value=-10))
    logB_interp_list.append(RegularGridInterpolator((np.log10(np.concatenate( (Bfiled_grid[i][::8, 0], Bfiled_grid[i][-1:, 0]) )), rad_bins_c),
                                                     np.log10(np.concatenate( (Bfiled_grid[i][::8, 3:], Bfiled_grid[i][-1:, 3:]) )),
                                                    bounds_error=False, fill_value=-10 ))

####### HALO MODEL ########

ellMax0   = 9600
ells      = np.arange(ellMax0)
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
rs  = np.linspace(rMin, rMax, 100000)            # halo radius
ks  = np.geomspace(1e-4,1e3, 5001)               # wavenumbers

print('Axion mass:', MA)
print('Halo masses:', mMin, mMax, nMs)
print('Redshifts:', zMin, zMax, nZs)

# Halo Model
dictnumber = 21
hod_name = "unWISE"+unWISEcol

hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir', concmode='BHATTACHARYA', unwise_color=unWISEcol, choose_dict=dictnumber)
hcos.add_hod(name=hod_name)
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
m200c, r200c = get_200critz(zs, ms, cs, rhocritz, deltav)

dvols  = get_volume_conv(chis, Hz)
PzkLin = hcos._get_matter_power(zs, ks, nonlinear=False)
Pzell  = get_fourier_to_multipole_Pkz(zs, ks, chis, ellMax0, PzkLin)
Pzell0 = Pzell.transpose(1,0)
print('Done turning into multipoles.')

path_params0 = np.asarray([MA, nZs, zMin, zMax, ellMax0, rscale])

dothis = False
if dothis:
    rcross = get_rcross_per_halo(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, pick_prof, name=name)
    ucosth, angs = get_halo_skyprofile(zs, chis, rcross)

    prob = conv_prob(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)
    #prob = conv_prob_flat(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)

    utheta = prob_theta(zs, ms, rs, rvirs, rhocritz, deltav, cs, MA, rcross, logB_interp_list, mass_bins, rad_bins, pick_prof, rscale=rscale, name=name)

    print('Computing multipole expansion of angular probability u.')
    partial_u = functools.partial(get_uell0, angs, ucosth)
    with ProcessPoolExecutor(num_workers) as executor:
        uell0 = list(executor.map(partial_u, ells, chunksize=chunksize))

    prob00 = prob * utheta * uell0[0]
    avtau, dtaudz = get_avtau(zs, ms, nzm, dvols, prob00)

    np.save(data_path(*path_params0), [rcross, prob, utheta, avtau, dtaudz, uell0])

    # For polarization, the number of resonances per halo is two, as L_domains < L_halo and in+out events are uncorrelated
    # so we need utheta**2.
    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
    probell_temp = (prob * utheta)[None,...] * uell0

    ct = np.sqrt((4.*np.pi)/(2*ells+1))
    zell_tau = (probell_temp[:ellMax0]) * ct[:, None, None]

    # Assemble power spectra
    int_uell_1h = np.trapz(nzm[None,...] * zell_tau**2.               , ms, axis=-1)
    int_uell_2h = np.trapz(nzm[None,...] * biases[None,...] * zell_tau, ms, axis=-1)

    Cl1h  = np.trapz(dvols[None,:] * int_uell_1h                     , zs, axis=1)
    Cl2h  = np.trapz(dvols[None,:] * np.abs(int_uell_2h)**2. * Pzell0, zs, axis=1)
    np.save(cl_data_tautau_path(*path_params0), [Cl1h, Cl2h])

    Cl1h, Cl2h = np.load(cl_data_tautau_path(*path_params0))
    np.save(ClTT0_path(*path_params0), (Cl1h + Cl2h) * TCMB**2.)
    

dothis=True
if dothis:
    for ellMax in [6500]:

        doPSorredBBg = False
        dopowc       = False
        dobisp2h     = False

        dobisp1h     = False

        dobispconstr = True
        doBBG1h      = False
        doBBG2h      = False
        doTTG1h_unwise = False
        doTTG1h_allcen = True

        l0Max, l1Max, l2Max = ellMax, ellMax, ellMax

        ells0 = np.arange(ellMax0)
        ells  = np.arange(ellMax)

        ellshort = np.array([0.] + np.linspace(2, ellMax-1, ellMax//12+1).tolist())
        ellshort = np.array([int(ii) for ii in ellshort])
        ellshort = np.array(list(dict.fromkeys(ellshort)))
        print(ellshort)

        ellshortshort = np.array([0.] + np.geomspace(2, ellMax-1, 101).tolist())
        ellshortshort = np.array([int(ii) for ii in ellshortshort])
        ellshortshort = np.array(list(dict.fromkeys(ellshortshort)))
        print(ellshortshort)

        path_params = np.asarray([MA, nZs, zMin, zMax, ellMax, rscale])

        # This is the unWISE survey template containing only central galaxys; function of z2 and m2
        Ncs  = hcos.hods[hod_name]['Nc']
        dndz = get_dndzHOD(zs, pathdndz, dvols)
        N_gtot = np.trapz(dndz, zs, axis=0)
        W_g = dndz / N_gtot / dvols
        ngalcentrals = np.trapz(nzm * Ncs[None, :], ms, axis=-1)
        ugcen = W_g[:, None] * Ncs[None, :] / ngalcentrals[:, None]

        if doPSorredBBg:
            for Npc in [5., 1., 10.]:
                print('Doing npc, MA', Npc, MA)

                # polarization window function; function of l1, z
                Clpol = get_gaussian_pol_1kp(zs, chis, ellMax0, Npc)

                # To compute the polarization screening, I am using the temperature dark screening integrand per halo
                # so I need to multiply by 3 in each halo
                # For polarization, the number of resonances per halo is one, as L_domains < L_halo and in+out events are uncorrelated
                rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
                ct = np.sqrt((4.*np.pi)/(2*ells0+1))
                probell_pol = 3. * (prob[None,...] * uell0)[:ellMax0,...] * ct[:, None, None]
                zell_pol    = utheta[None,...] * np.abs(probell_pol)**2.

                if dopowc:
                    # one-halo axion screening; function of l2, z
                    integrBB = dvols[None, :] * np.trapz(nzm[None,...] * zell_pol, ms, axis=-1)

                    Clmix = np.trapz(integrBB[:ellMax, None, :] * Clpol[None, :ellMax, :], zs, axis=-1)
                    scrEE = get_scrCLs_pol_axion(l0Max, l1Max, l2Max, ellshortshort, Clmix, TCMB)

                    np.save(fullscr_polaxion_tautau_path(*path_params, Npc), scrEE)
                    print('Saved BB auto MA, npc', MA, Npc)

                if dobisp2h:
                    # The dark screening integrand; function of L and z
                    integrBB = np.trapz( (nzm * biases)[None,...] * zell_pol, ms, axis=-1)

                    # The central gal integrand times the linear matter power; function of ell'' and z
                    integrCen = np.trapz(nzm * biases * ugcen, ms, axis=-1)
                    dCellhh   = (dvols * integrCen)[None,:] * Pzell0

                    # multiplying by magnetic field domain shape; integrate z away; shape ell'', l_1, l_2
                    integr = dCellhh[ellshortshort, None, None, :] * integrBB[None, :ellMax, None, :] * Clpol[None, None, ellshortshort, :]
                    Blitgr = np.trapz(integr, zs, axis=-1)

                    # Keeping ell'' fixed, sum over l1, l2 weighting by appropriate 3j symbols
                    # It returns the linear matter PS ell on position 0 and the other ell on position 1
                    Bispec = get_scrBLs_polaxion_gal(l0Max, l1Max, l2Max, ellshortshort, Blitgr, TCMB)

                    np.save(fullscr_reducedbisp_tautau_path(*path_params, Npc) + '_unWISE.npy', Bispec)
                    print('Saved 2h bisp MA, npc', MA, Npc)


        if dobisp1h:
            for Npc in [5.]:#, 1., 10.]:
                print('Doing npc, MA', Npc, MA)

                rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
                ct = np.sqrt((4.*np.pi)/(2*ells+1))
                probell_pol = 3. * (prob[None,...] * uell0)[:ellMax,...] * ct[:, None, None]

                # Integrate over mass, do not integrate over redshift
                # this is a function of L, L' and z
                probcross = probell_pol[ellshort,None,:,:] * probell_pol[None,ellshort,:,:]
                integrBB = np.trapz( (nzm * utheta * ugcen)[None,None,:,:] * probcross, ms, axis=-1)

                # Keeping z, sum over L, L' weighting by appropriate 3j symbols
                # It returns the free index l'' on position 0, z on position 1
                Blitgr = get_scrBLs_1h_polaxion_gal(l0Max, l1Max, l2Max, ellshort, integrBB, TCMB)

                # polarization window function; function of l1, z
                Clpol = get_gaussian_pol_1kp(zs, chis, ellMax, Npc)

                # don't forget volume factor and polarization window
                Bispec = Blitgr[None,...] * Clpol[ellshortshort,None,:]
                Bispec = np.trapz(dvols[None,None,:] * Bispec, zs, axis=-1)
                # so now, index 0 carries polarization, index 1 carries galaxy template

                np.save(fullscr_1h_reducedbisp_tautau_path(*path_params, Npc) + '_unWISE.npy', Bispec)
                print('Saved 1h bisp MA, npc', MA, Npc)

        if dobispconstr:
            for eid, (expname, experiment) in enumerate(zip(['Planck', 'CMBS4'], [Planck, CMBS4])):
                if eid!=1: continue

                baseline = ghztoev(145)
                units  = xov(baseline) * baseline
                fsky = 0.4

                if expname == 'Planck':
                    mm, mmm = 4, 3000
                elif expname == 'CMBS4':
                    mm, mmm = 4, 6000

                path = ILCnoisePS_path(MA, nZs, zMin, zreio, ellMax0, expname)
                leftover = np.load(path)

                path = cl_data_galgal_path(nZs, zMin, zreio, ellMax0, name=name, galcol=unWISEcol, dictn=dictnumber) + 'centrals_only_unWISE.npy'
                Cell_galgal = np.load(path)

                if doBBG1h:
                    for Npc in [5.]:#[5., 1., 10.]:
                        print('Doing', expname, Npc, MA)
                        path   = fullscr_1h_reducedbisp_tautau_path(MA, nZs, zMin, zreio, ellMax, rscale, Npc)+'_unWISE.npy'
                        Bispec = np.load(path)

                        # index 0 carries polarization, index 1 carries galaxy template
                        xg, yg = np.meshgrid(ells, ells, indexing='ij', sparse=True)
                        f      = RegularGridInterpolator((ellshortshort, ells), Bispec)
                        Bispec = f((xg, yg)) * units**2.

                        constr = get_1h_bispectrum_constraint(fsky, mm, mmm, leftover[2], Cell_galgal, Bispec)
                        np.save(BBg_1h_bispectrum_constraint(MA, nZs, zMin, zreio, mmm, Npc, expname, unwi=True), constr)
                        print('Saved BBg 1h constraint', expname, Npc, MA, constr)

                if doBBG2h:
                    for Npc in [5., 1., 10.]:
                        print('Doing', expname, Npc, MA)
                        path   = fullscr_reducedbisp_tautau_path(MA, nZs, zMin, zreio, ellMax, rscale, Npc)+'_unWISE.npy'
                        Bispec = np.load(path)

                        xg, yg = np.meshgrid(ells, ells, indexing='ij', sparse=True)
                        f      = RegularGridInterpolator((ellshortshort, ells), Bispec)
                        Bispec = f((xg, yg)) * units**2.

                        constr = get_bispectrum_constraint(fsky, mm, mmm, leftover[2], Cell_galgal, Bispec)
                        np.save(bispectrum_constraint(MA, nZs, zMin, zreio, mmm, Npc, expname, unwi=True), constr)
                        print('Saved BBg 2h constraint', expname, Npc, MA, constr)

                if doTTG1h_unwise:
                    print('Doing', expname, MA)
                    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
                    probell_temp = (prob * utheta)[None,...] * uell0

                    ct = np.sqrt((4.*np.pi)/(2.*ells+1.))
                    zell_tau = probell_temp[:ellMax] * ct[:, None, None]
                    zell_tau_cross = zell_tau[None,ellshort,:,:] * zell_tau[ellshort,None,:,:]

                    integr = np.trapz( (nzm * ugcen)[None,None,...] * zell_tau_cross, ms, axis=-1)
                    bispecTTg1h = np.trapz(dvols[None,None,:] * integr, zs, axis=-1)

                    xg, yg = np.meshgrid(ells, ells, indexing='ij', sparse=True)
                    f = RegularGridInterpolator((ellshort, ellshort), bispecTTg1h)
                    Bispec = f((xg, yg)) * TCMB**2. * units**2.

                    constr = get_TTg_bispectrum_constraint(fsky, mm, mmm, leftover[0], Cell_galgal, Bispec)

                    np.save(TTg_1h_bispectrum_constraint(MA, nZs, zMin, zreio, mmm, expname, unwi=True), constr)
                    print('Saved TTg 1h constraint', expname, MA, constr)

                if doTTG1h_allcen:
                    # Every halo in the halo mass function has one central; function of z2 and m2
                    Ncs  = np.ones(nMs)
                    dndz = np.ones(nZs)
                    N_gtot = np.trapz(dndz, zs, axis=0)
                    W_g = dndz / N_gtot / dvols
                    ngalcentrals = np.trapz(nzm * Ncs[None,:], ms, axis=-1)
                    ugcen = W_g[:,None] * Ncs[None,:] / ngalcentrals[:,None]

                    path = cl_data_galgal_path(nZs, zMin, zreio, ellMax0, name=name, galcol=unWISEcol, dictn=dictnumber) + 'centrals_only.npy'
                    Cell_galgal = np.load(path)

                    print('Doing', expname, MA)
                    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*path_params0))
                    probell_temp = (prob * utheta)[None,...] * uell0

                    ct = np.sqrt((4.*np.pi)/(2.*ells+1.))
                    zell_tau = probell_temp[:ellMax] * ct[:, None, None]
                    zell_tau_cross = zell_tau[None,ellshort,:,:] * zell_tau[ellshort,None,:,:]

                    integr = np.trapz( (nzm * ugcen)[None,None,...] * zell_tau_cross, ms, axis=-1)
                    bispecTTg1h = np.trapz(dvols[None,None,:] * integr, zs, axis=-1)

                    xg, yg = np.meshgrid(ells, ells, indexing='ij', sparse=True)
                    f      = RegularGridInterpolator((ellshort, ellshort), bispecTTg1h)
                    Bispec = f((xg, yg)) * TCMB**2. * units**2.

                    constr = get_TTg_bispectrum_constraint(fsky, mm, mmm, leftover[0], Cell_galgal, Bispec)

                    np.save(TTg_1h_bispectrum_constraint(MA, nZs, zMin, zreio, mmm, expname, unwi=True)+'all_centrals.npy', constr)
                    print('Saved TTg 1h constraint', expname, MA, constr)

print('Done maind, MA', maind, MA)
