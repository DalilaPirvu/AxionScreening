import numpy as np

############# Define constants ################
# From hmvec:
# proper radius r is always in Mpc
# comoving momentum k is always in Mpc-1
# All masses m are in Msolar
# rho densities are in Msolar/Mpc^3
# No h units anywhere

############# Define constants ################
# From hmvec:
# proper radius r is always in Mpc
# comoving momentum k is always in Mpc-1
# All masses m are in Msolar
# rho densities are in Msolar/Mpc^3
# No h units anywhere

TFIRAS   = 2.725     # in Kelvin
TCMB     = 2.726*1e6 # in micro Kelvin
cmMpc    = 3.2407792896e-25    # Mpc/cm            # how many Mpc in a cm
eVinvCm  = 1.97*1e-5  #1.2398419e-4        # cm/eV^-1          # how many cm in a eV^-1
mpcEVinv = 1./(cmMpc*eVinvCm)  # eV^-1/Mpc         # how many eV^-1 in a Mpc

mMWvir  = 1.3*1e12
rMWvir  = 287*1e-3
rEarth  = 8*1e-3
csMW = 10.72 # log10 c = 1.025 - 0.097 log10 (M / (10^12 /h solar masses)) # h = 0.68

msun   = 1.9891e30     # kg               # Sun mass
mprot  = 1.67262e-27   # kg               # Proton mass
m2eV   = 1.4e-21       # eV^2             # conversion factor for plasma mass (eq. (2) in Caputo et al; PRL)
ombh2  = 0.02225                 # Physical baryon density parameter Ωb h2
omch2  = 0.1198                  # Physical dark matter density parameter Ωc h2
conv   = m2eV*(ombh2/omch2)*(msun/mprot)*(cmMpc)**3.

thomson = 0.6652*1e-24
conv2 = thomson*(ombh2/omch2)*(msun/mprot)*(cmMpc)**2.

kB      = 8.61732814974493*1e-5   # Boltzmann constant in eV/Kelvin
K2eV    = lambda K: kB * K
cligth  = 299792458.0             # m/s
clight1 = 9.71561e-15             # Mpc/s

hplanck    = 6.62607015*1e-34     # not hbar!            # m2 kg / s     = Joule * second
kboltzmann = 1.380649*1e-23                              # m2 kg s-2 K-1 = Joule / Kelvin
xx0  = lambda nu: hplanck * nu*1e9 / kboltzmann / TFIRAS # for nu in GHz
xov0 = lambda nu: (1. - np.exp(-xx0(nu))) / xx0(nu)

xx  = lambda om: om / (kB * TFIRAS)      # for omega in eV
xov = lambda om: (1. - np.exp(-xx(om))) / xx(om)

frq = lambda nu: 100. * nu * cligth
BBf = lambda frq: 1e26/1e6 * (2.*frq**3.*hplanck)/cligth**2. / (np.exp(frq * hplanck/kboltzmann/TFIRAS) - 1.)

BBω = lambda omg: (omg**3.)/(2.*np.pi**2.) / (np.exp(omg/K2eV(TFIRAS)) - 1.)

aa = lambda z: 1./(1.+z)

arcmin2rad = lambda arcm: arcm/60. * np.pi/180.
ghztoev    = lambda GHz: 4.13566553853809E-06 * GHz
gauss2evsq = lambda gauss: 1.95e-2 * gauss

############# Halo models ############# 

dictKey1 = np.asarray([1.4e-13, 1.5e-13, 1.6e-13, 1.7e-13, 1.8e-13, 1.9e-13, \
                      2.0e-13, 2.1e-13, 2.2e-13, 2.3e-13, 2.4e-13, 2.5e-13, 2.6e-13, 2.7e-13, 2.8e-13, 2.9e-13, \
                      3.0e-13, 3.1e-13, 3.2e-13, 3.3e-13, 3.4e-13, 3.5e-13, 3.6e-13, 3.7e-13, 3.8e-13, 3.9e-13, \
                      4.0e-13, 4.1e-13, 4.2e-13, 4.3e-13, 4.4e-13, 4.5e-13, 4.6e-13, 4.7e-13, 4.8e-13, 4.9e-13, \
                      5.0e-13, 5.1e-13, 5.2e-13, 5.3e-13, 5.4e-13, 5.5e-13, 5.6e-13, 5.7e-13, 5.8e-13, 5.9e-13, \
                      6.0e-13, 6.1e-13, 6.2e-13, 6.4e-13, 6.6e-13, 6.8e-13, \
                      7.0e-13, 7.2e-13, 7.4e-13, 7.6e-13, 7.8e-13, \
                      8.0e-13, 8.2e-13, 8.4e-13, 8.6e-13, 8.8e-13, \
                      9.0e-13, 9.2e-13, 9.4e-13, 9.6e-13, 9.8e-13, \
                      1.0e-12, 1.2e-12, 1.4e-12, 1.6e-12, 1.8e-12, \
                      2.0e-12, 2.3e-12, 2.7e-12, \
                      3.0e-12, 3.3e-12, 3.7e-12, \
                      4.0e-12, 4.3e-12, 4.7e-12, \
                      5.0e-12, 5.3e-12, 5.7e-12, \
                      6.0e-12, 6.5e-12, \
                      7.0e-12, 7.5e-12, \
                      8.0e-12, 8.5e-12, \
                      9.0e-12, 9.5e-12, \
                      1.0e-11, 1.5e-11, \
                      2.0e-11])

dictKey = np.asarray([1.4e-13, 1.5e-13, 1.6e-13, 1.7e-13, 1.9e-13, \
                      2.0e-13, 2.2e-13, 2.4e-13, 2.6e-13, 2.8e-13, \
                      3.0e-13, 3.2e-13, 3.4e-13, 3.6e-13, 3.8e-13, \
                      4.0e-13, 4.2e-13, 4.4e-13, 4.6e-13, 4.8e-13, \
                      5.0e-13, 5.2e-13, 5.4e-13, 5.6e-13, 5.8e-13, \
                      6.0e-13, 6.2e-13, 6.4e-13, 6.6e-13, 6.8e-13, \
                      7.0e-13, 7.4e-13, 7.8e-13, 8.0e-13, 8.4e-13, 8.8e-13, \
                      9.0e-13, 9.4e-13, 9.8e-13, 1.0e-12, 1.4e-12, 1.8e-12, \
                      2.0e-12, 2.3e-12, 2.7e-12, 3.0e-12, 3.3e-12, 3.7e-12, \
                      4.0e-12, 4.3e-12, 4.7e-12, 5.0e-12, 5.3e-12, 5.7e-12, \
                      6.0e-12, 6.5e-12, 7.0e-12, 7.5e-12, 8.0e-12, 8.5e-12, 9.0e-12, 9.5e-12, 1.0e-11, 1.5e-11, 2.0e-11])

modelParams = {1.4e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               1.5e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               1.6e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               1.7e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               1.9e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               2.0e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               2.2e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               2.4e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               2.6e-13: np.asarray([0.005, 2.,  1e-6, 1e2]),\
               2.8e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\

               3.0e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\
               3.2e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\
               3.4e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\
               3.6e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\
               3.8e-13: np.asarray([0.005, 3.,  1e-6, 1e2]),\
               4.0e-13: np.asarray([0.005, 4.,  1e-6, 5e1]),\
               4.2e-13: np.asarray([0.005, 4.,  1e-6, 5e1]),\
               4.4e-13: np.asarray([0.005, 4.,  1e-6, 5e1]),\
               4.6e-13: np.asarray([0.005, 4.,  1e-6, 5e1]),\
               4.8e-13: np.asarray([0.005, 5.,  1e-6, 5e1]),\
               
               5.0e-13: np.asarray([0.005, 5.,  1e-6, 5e1]),\
               5.2e-13: np.asarray([0.005, 5.,  1e-6, 5e1]),\
               5.4e-13: np.asarray([0.005, 5.,  1e-6, 5e1]),\
               5.6e-13: np.asarray([0.005, 5.,  1e-6, 5e1]),\
               5.8e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               6.0e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               6.2e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               6.4e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               6.6e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               6.8e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\

               7.0e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               7.4e-13: np.asarray([0.005, 6.,  1e-6, 5e1]),\
               7.8e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               8.0e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               8.4e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               8.8e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               9.0e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               9.4e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               9.8e-13: np.asarray([0.005, 10., 1e-6, 5e1]),\
               1.0e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\

               1.4e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\
               1.8e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\
               2.0e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\
               2.3e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\
               2.7e-12: np.asarray([0.005, 10., 1e-6, 5e1]),\
               3.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               3.3e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               3.7e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               4.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               4.3e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\

               4.7e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               5.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               5.3e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               5.7e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               6.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               6.5e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               7.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               7.5e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               8.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               8.5e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\

               9.0e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               9.5e-12: np.asarray([0.005, 10., 1e-6, 1e1]),\
               1.0e-11: np.asarray([0.005, 10., 1e-6, 1e1]),\
               1.5e-11: np.asarray([0.005, 10., 1e-6, 1e1]),\
               2.0e-11: np.asarray([0.005, 10., 1e-6, 1e1])}

def chooseModel(chosenMass, models):
    try:
        return models[chosenMass]
    except:
        print('Mass not implemented.')

def import_data(MA, nZs, zMin, zMax, ellMax0, npc, ellMax, unwise=False, rscale=False):
    params = np.array([MA, nZs, zMin, zMax, ellMax0, rscale])
    rcross, prob, utheta, avtau, dtaudz, uell0 = np.load(data_path(*params))

    Cell1Halo, Cell2Halo = np.load(cl_data_tautau_path(*params))

    scrTT = (Cell1Halo + Cell2Halo) * TCMB**2.

    params = np.array([MA, nZs, zMin, zMax, ellMax, rscale, npc])

    # Axion polarization 2-point function
    scrEE = np.load(fullscr_polaxion_tautau_path(*params))
    CMBDP = np.array([scrTT[:ellMax], scrEE, scrEE])
    return rcross, prob, utheta, avtau, dtaudz, uell0, Cell1Halo, Cell2Halo, CMBDP


############# Noise modelling ############# 

Planck = {'freqsGHz'     :            np.array([30,     44,   70,     100,  143, 217,  353,  545,  857  ]) ,\
          'freqseV'      :    ghztoev(np.array([30,     44,   70,     100,  143, 217,  353,  545,  857  ])),\
          'FWHMarcmin'   :            np.array([32.408, 27.1, 13.315, 9.69, 7.3, 5.02, 4.94, 4.83, 4.64 ]) ,\
          'FWHMrad'      : arcmin2rad(np.array([32.408, 27.1, 13.315, 9.69, 7.3, 5.02, 4.94, 4.83, 4.64 ])),\
          'SensitivityμK':            np.array([195.1, 226.1, 199.1, 77.4, 33., 46.8, 153.6, 818.2, 40090.7])*arcmin2rad(1.),\
          'Knee ell'     : 0.,\
          'Exponent'     : 0.}

CMBS4 = {'freqsGHz'  :            np.array([20,    27,   39,   93,   145,  225,  278 ]) ,\
         'freqseV'   :    ghztoev(np.array([20,    27,   39,   93,   145,  225,  278 ])),\
         'FWHMarcmin':            np.array([11.0,  8.4,  5.8,  2.5,  1.6,  1.1,  1.0 ]) ,\
         'FWHMrad'   : arcmin2rad(np.array([11.0,  8.4,  5.8,  2.5,  1.6,  1.1,  1.0 ])),\
         'SensitivityμK':         np.array([10.41, 5.14, 3.28, 0.50, 0.46, 1.45, 3.43])*arcmin2rad(1.) ,\
         'Knee ell'  : 100.,\
         'Exponent'  : -3.}

CMBHD = {'freqsGHz'     :            np.array([30,   40,   90,   150,  220,  280,  350])  ,\
         'freqseV'      :    ghztoev(np.array([30,   40,   90,   150,  220,  280,  350])) ,\
         'FWHMarcmin'   :            np.array([1.25, 0.94, 0.42, 0.25, 0.17, 0.13, 0.11] ),\
         'FWHMrad'      : arcmin2rad(np.array([1.25, 0.94, 0.42, 0.25, 0.17, 0.13, 0.11])),\
         'SensitivityμK':            np.array([6.5,  3.4,  0.7,  0.8,  2.0,  2.7,  100.0])*arcmin2rad(1.),\
         'Knee ell'     : 100.,\
         'Exponent'     : -3.}

#############  Storage #############

dirplots   = './plots/'
dirsomedata= './data/'

#dirdata = lambda MA,nZs,zMin,zMax,lMax: '/gpfs/dpirvu/axiondata/MA%5.4e'%(MA) + '_nZs'+str(int(nZs)) + '_zmin'+str(zMin) + '_zmax'+str(zMax) + '_ellMax'+str(int(lMax))
#dirdata_thom = lambda nZs,zMin,zMax,lMax: '/gpfs/dpirvu/axiondata/nZs'+str(int(nZs)) + '_zmin'+str(zMin) + '_zmax'+str(zMax) + '_ellMax'+str(int(lMax))

dirdata = lambda MA,nZs,zMin,zMax,lMax: '/gpfs/dpirvu/axiondata/Limber_MA%5.4e'%(MA) + '_nZs'+str(int(nZs)) + '_zmin'+str(zMin) + '_zmax'+str(zMax) + '_ellMax'+str(int(lMax))
dirdata_thom = lambda nZs,zMin,zMax,lMax: '/gpfs/dpirvu/axiondata/Limber_nZs'+str(int(nZs)) + '_zmin'+str(zMin) + '_zmax'+str(zMax) + '_ellMax'+str(int(lMax))

data_path = lambda MA,nZs,zMin,zMax,lMax,rscale: dirdata(MA,nZs,zMin,zMax,lMax) + '_files_'  + ('rscale' if rscale else 'r0') + '.npy'
ClTT0_path= lambda MA,nZs,zMin,zMax,lMax,rscale: dirdata(MA,nZs,zMin,zMax,lMax) + '_CMBDP0_' + ('rscale' if rscale else 'r0') + '.npy'
ClEE_path= lambda MA,nZs,zMin,zMax,lMax,rscale: dirdata(MA,nZs,zMin,zMax,lMax) + '_CMBDP_pol_' + ('rscale' if rscale else 'r0') + '.npy'

cl_data_tautau_path = lambda MA,nZs,zMin,zMax,lMax,rscale: dirdata(MA,nZs,zMin,zMax,lMax) + '_tautau_Cl_' + ('rscale' if rscale else 'r0') + '.npy'
fullscr_tautau_path = lambda MA,nZs,zMin,zMax,lMax,rscale: dirdata(MA,nZs,zMin,zMax,lMax) + '_tautau_CMBDP_' + ('rscale' if rscale else 'r0') + '.npy'

fullscr_polaxion_tautau_path        = lambda MA,nZs,zMin,zMax,lMax,rscale,npc: dirdata(MA,nZs,zMin,zMax,lMax)+'_fullpolscr_npc'  +str(int(npc))+'_tautau_CMBDP_'+('rscale' if rscale else 'r0')+'.npy'
fullscr_reducedbisp_tautau_path     = lambda MA,nZs,zMin,zMax,lMax,rscale,npc: dirdata(MA,nZs,zMin,zMax,lMax)+'_reducedbipsl_npc'+str(int(npc))+'_tautau_CMBDP_'+('rscale' if rscale else 'r0')+'.npy'
fullscr_1h_reducedbisp_tautau_path  = lambda MA,nZs,zMin,zMax,lMax,rscale,npc: dirdata(MA,nZs,zMin,zMax,lMax)+'_1h_reducedbipsl_npc'+str(int(npc))+'_tautau_CMBDP_'+('rscale' if rscale else 'r0')+'.npy'

cl_data_galgal_path = lambda nZs,zMin,zMax,lMax,name,galcol,dictn: dirdata_thom(nZs,zMin,zMax,lMax) +name+'_galgal_Cl_'+galcol+str(dictn)+'.npy'
cl_data_galtau_path = lambda MA,nZs,zMin,zMax,lMax,name,galcol,dictn: dirdata(MA,nZs,zMin,zMax,lMax)+name+'_galtau_Cl_'+galcol+str(dictn)+'.npy'

ILCnoisePS_path = lambda MA,nZs,zMin,zMax,lMax,expname: dirdata(MA, nZs, zMin, zMax,lMax) + 'ILC_remainder_' + expname + '.npy'
weights_path    = lambda MA,nZs,zMin,zMax,lMax,expname: dirdata(MA, nZs, zMin, zMax,lMax) + 'ILC_weights_'   + expname + '.npy'

bispectrum_constraint = lambda MA,nZs,zMin,zMax,lMax,npc,expname,unwi: dirdata(MA,nZs,zMin,zMax,lMax)+'bispectra_constraints_npc'+str(int(npc))+expname+('_unWISE' if unwi else '_non')+'.npy'
TTg_1h_bispectrum_constraint = lambda MA,nZs,zMin,zMax,lMax,expname,unwi: dirdata(MA,nZs,zMin,zMax,lMax)+'bispectrumTTg1h_constraints'+expname+('_unWISE' if unwi else '_non')+'.npy'
BBg_1h_bispectrum_constraint = lambda MA,nZs,zMin,zMax,lMax,npc,expname,unwi: dirdata(MA,nZs,zMin,zMax,lMax)+'bispectrumBBg1h_constraints_npc'+str(int(npc))+expname+('_unWISE' if unwi else '_non')+'.npy'

