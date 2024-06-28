import os,sys
sys.path.append('./hmvec-master/')
import hmvec as hm # Git clone and pip install as in readme from github.com/msyriac/hmvec
from compute_power_spectra import *
from plotting import *
from params import *

np_load_old = np.load
np.load     = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


ellMax = 9800
ells = np.arange(ellMax)

getgas = True
dictKey = dictKey[:40]
model = modelParams
rscale = False

cych = ['#377eb8', '#ff7f00', 'forestgreen', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

baseline = ghztoev(30)
units = baseline * mpcEVinv * gauss2evsq(1e-7)**2.
print(units)

ztype = [2., 6.]
zreio = 6.
nZs = 50

fsky = [0.7, 0.5, 0.5]
pick_prof = True

name = 'battagliaAGN'
rscale = False

rs  = np.linspace(1e-6, 1e2,10000)              # halo radius

avtaulist = np.zeros((len(ztype), len(dictKey)))
dtaudzlist, zsList = np.zeros((2, len(ztype), len(dictKey), 50))
Cell1Hdata, Cell2Hdata, CellTauTau = np.zeros((3, len(ztype), len(dictKey), ellMax))
Screening = np.zeros((len(ztype), len(dictKey), 4, ellMax))
Survey = np.zeros((len(dictKey), ellMax))
rcrossdata, probdata = np.zeros((2, len(ztype), len(dictKey), 50, 100))

for mind, MA in enumerate(dictKey):
    for zind, ztest in enumerate(ztype):

        zMin, zMax, rMin, rMax = chooseModel(MA, model)
        zMax = min(ztest, zMax)

        data = import_data_short(MA, nZs, zMin, zMax, ellMax, getgas, rscale)
        rcross, prob, avtau, dtaudz, uell0, Cell1H, Cell2H, CMBDP  = data

        rcrossdata[zind, mind] = rcross
        probdata[zind, mind]   = prob

        avtaulist[zind, mind]  = avtau * units
        dtaudzlist[zind, mind] = dtaudz * units

        zsList[zind, mind]     = np.linspace(zMin,zMax,nZs)

        Cell1Hdata[zind, mind] = Cell1H
        Cell2Hdata[zind, mind] = Cell2H

        CellTauTau[zind, mind] = (Cell1H + Cell2H) * units**2.
        Screening[zind, mind]  = CMBDP * units**2.

for mind, MA in enumerate(dictKey):
    Survey[mind] = (Cell1Hdata[0, mind] + Cell2Hdata[0, mind])


if True:
    fullzs = np.linspace(0.005, 6., 50)
    partzs = [fullzs[0], fullzs[8], fullzs[16], fullzs[25], fullzs[-1]]
    lists  = np.array([[ii, jj] for ii, jj in zip(partzs[:-1], partzs[1:])])
    mlists = np.array([[mind, MA] for mind, MA in enumerate(dictKey)])

    ms = np.geomspace(1e11,1e17,100)         # masses
    ks = np.geomspace(1e-4,1e3,101)          # wavenumbers

    xdat, ydat = np.ones((2, len(lists), len(mlists)))
    for a1, (zmin, zmax) in enumerate(lists):
        print(zmin, zmax)

        for a2, (mind, MA) in enumerate(mlists):
            mind = int(mind)
            print(mind, MA)

            # the redshifts relevant for conversion of dark photon of fixed mass m_A
            zsss = zsList[1, mind, :]

            zs    = np.linspace(zsss[0],zsss[-1],50)    # redshifts
            hcos  = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', mdef='vir')
            nzm   = hcos.get_nzm()

            # the physical redshift range of interest
            zi = (zs>=zmin) & (zs<zmax)

            # total number of halos within redshift range
            Ntot = np.trapz(np.trapz(nzm[zi, :], ms, axis=-1), zs[zi], axis=0)

            # average r_cross weighted by the number density
            ydat[a1, a2] = np.trapz(np.trapz(rcrossdata[1, mind, zi, :] * nzm[zi, :], ms, axis=-1), zs[zi], axis=0) / Ntot
            xdat[a1, a2] = dictKey[mind]

    np.save('./data/rresdat.npy', [lists, mlists, xdat, ydat])
