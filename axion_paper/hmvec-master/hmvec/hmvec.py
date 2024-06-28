import sys,os
import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d
import camb
from camb import model
import numpy as np
from . import tinker,utils
from .cosmology import Cosmology

import scipy.constants as constants
from .params import default_params, battaglia_defaults
from .fft import generic_profile_fft
import scipy
from scipy.integrate import simps

"""

General vectorized FFT-based halo model implementation
Author(s): Mathew Madhavacheril
Credits: Follows approach in Matt Johnson and Moritz
Munchmeyer's implementation in the appendix of 1810.13423.
Some of the HOD functions are copied from there.

Array indexing is as follows:
[z,M,k/r]

r is always in Mpc
k is always in Mpc-1
All rho densities are in Msolar/Mpc^3
All masses m are in Msolar
No h units anywhere

TODO: copy member functions like Duffy concentration to independent
barebones functions in separate script/library

Known issues:
1. The 1-halo term will add some power at the largest scales. A softening term
has been added that effectively zeros the 1-halo term below k<0.01.
2. sigma2 becomes very innacurate for low masses, because lower masses require higher
k in the linear matter power. For this reason, low-mass halos
are never accounted for correctly, and thus consistency relations do not hold.
Currently the consistency relation is subtracted out to get the 2-halo
power to agree with Plin at large scales.
3. Higher redshifts have less than expected 1-halo power compared to halofit. 

Limitations:
1. Tinker 2010 option and Sheth-Torman have only been coded up for M200m and mvir
respectively.

 In Fisher calculations, I want to calculate power spectra at a fiducial parameter set      
 and at perturbed parameters (partial derivatives). The usual flowdown for a power      
 spectrum calculation is:       
 C1. initialize background cosmology and linear matter power        
 C2. calculate mass function        
 C3. calculate profiles and HODs        
 with each step depending on the previous. This means if I change a parameter associated        
 with C3, I don't need to recalculate C1 and C2. So a Fisher calculation with n-point derivatives       
 should be      
 done based on a parameter set p = {p1,p2,p3} for each of the above as follows:     
 1. Calculate power spectra for fiducial p (1 C1,C2,C3 call each)       
 2. Calculate power spectra for perturbed p3 (n C3 calls for each p3 parameter)     
 3. Calculate power spectra for perturbed p2 (n C2,C3 calls for each p2 parameter)      
 2. Calculate power spectra for perturbed p1 (n C1,C2,C3 calls for each p1 parameter)       
        
 """

def Wkr_taylor(kR):
    xx = kR*kR
    return 1 - .1*xx + .00357142857143*xx*xx

def Wkr(k,R,taylor_switch=default_params['Wkr_taylor_switch']):
    kR = k*R
    ans = 3.*(np.sin(kR)-kR*np.cos(kR))/(kR**3.)
    ans[kR<taylor_switch] = Wkr_taylor(kR[kR<taylor_switch]) 
    return ans

def duffy_concentration(m,z,A=None,alpha=None,beta=None,h=None):
    A = default_params['duffy_A_mean'] if A is None else A
    alpha = default_params['duffy_alpha_mean'] if alpha is None else alpha
    beta = default_params['duffy_beta_mean'] if beta is None else beta
    h = default_params['H0'] / 100. if h is None else h
    return A*((h*m/2.e12)**alpha)*(1+z)**beta

def BHATTACHARYA_concentration(m, nus, Dz, h=None):
    h = default_params['H0'] / 100. if h is None else h
    #nu2m = lambda m: (1./Dz) * ( 1.12 * (h*m/5.e13)**0.3 + 0.53 )
    #return Dz**0.9 * 7.7 * nu2m(m)**-0.29
    return Dz**0.9 * 7.7 * nus**-0.29

def BHATTACHARYA_concentration200(m, nus, Dz, h=None):
    h = default_params['H0'] / 100. if h is None else h
    #nu2m = lambda m: (1./Dz) * ( 1.12 * (h*m/5.e13)**0.3 + 0.53 )
    #return Dz**0.54 * 5.9 * nu2m(m)**-0.35
    return Dz**0.54 * 5.9 * nus**-0.35

class HaloModel(Cosmology):
    def __init__(self,zs,ks,ms=None,params=None,mass_function="sheth-torman", unwise_color="blue", choose_dict=0,
                 halofit=None,mdef='vir',concmode='BHATTACHARYA',nfw_numeric=False,skip_nfw=False,accurate_sigma2=False):
        self.zs = np.asarray(zs)
        self.ks = ks
        self.accurate_sigma2 = accurate_sigma2
        Cosmology.__init__(self,params,halofit)

        self.mdef = mdef
        self.mode = mass_function
        self.unWISE = unwise_color
        self.chooseHOD = choose_dict
        self.conc = concmode
        self.hods = {}

        if self.unWISE == 'blue':
            hod_params = blue_dicts[self.chooseHOD]
        elif self.unWISE == 'green':
            hod_params = green_dicts[self.chooseHOD]

        lamb = hod_params['hod_lambda']
        print('unwise_color', self.unWISE, 'choose_dict', self.chooseHOD, 'hod_params', hod_params)

        # Mass function
        if ms is not None: 
            self.ms = np.asarray(ms)
            self.init_mass_function(self.ms)

        # Profiles
        self.uk_profiles = {}
        self.pk_profiles = {}
        if not(skip_nfw):
          #  self.add_nfw_profile("nfw",numeric=nfw_numeric)
            self.add_nfw_profile_new(lamb, "nfw")

        
    def _init_cosmology(self,params,halofit):
        Cosmology._init_cosmology(self,params,halofit)
        self.Pzk = self._get_matter_power(self.zs,self.ks,nonlinear=False)
        if halofit is not None: self.nPzk = self._get_matter_power(self.zs,self.ks,nonlinear=True)

    def deltav(self,z): # Duffy virial actually uses this from Bryan and Norman 1997
        # return 178. * self.omz(z)**(0.45) # Eke et al 1998
        x = self.omz(z) - 1.
        d = 18.*np.pi**2. + 82.*x - 39. * x**2.
        return d

    def rvir(self,m,z):
        if self.mdef == 'vir':
            return R_from_M(m,self.rho_critical_z(z),delta=self.deltav(z))
        elif self.mdef == 'mean':
            return R_from_M(m,self.rho_matter_z(z),delta=200.)

    def mrc200(self):
        # Convert masses to m200critz
        zs = self.zs
        ms = self.ms
        rhocritz = self.rho_critical_z(zs)
        if self.mdef=='vir':
            delta_rhos1 = rhocritz*self.deltav(zs)
        elif self.mdef=='mean':
            delta_rhos1 = self.rho_matter_z(zs)*200.
        cs = self.concentration()
        rvirs = self.rvir(ms[None,:],zs[:,None])
        delta_rhos2 = 200.*self.rho_critical_z(zs)
        m200critz = mdelta_from_mdelta(ms,cs,delta_rhos1,delta_rhos2)
        r200critz = R_from_M(m200critz,self.rho_critical_z(zs)[:,None],delta=200.)

        if self.conc=='duffy':
            logMdiff = np.log(m200critz) - np.log(ms)
            c200critz = cs * (np.exp(logMdiff) * (delta_rhos1/delta_rhos2)[:,None]) **(1./3.)

        elif self.conc=='BHATTACHARYA':
            sigma2 = self.sigma2
            deltac = self.p['st_deltac']
            nus = deltac/np.sqrt(sigma2)
            Dzs = self.D_growth(z2a(zs))
            c200critz = BHATTACHARYA_concentration200(m200critz, nus, Dzs[:,None], self.h)

        return m200critz, r200critz, c200critz
    
    def R_of_m(self,ms):
        return R_from_M(ms,self.rho_matter_z(0),delta=1.) # note rhom0

    def get_sigma2(self):
        ms = self.ms
        kmin = self.p['sigma2_kmin']
        kmax = self.p['sigma2_kmax']
        numks = self.p['sigma2_numks']
        self.ks_sigma2 = np.geomspace(kmin,kmax,numks) # ks for sigma2 integral
        if self.accurate_sigma2:
            self.sPzk = self.P_lin_slow(self.ks_sigma2,self.zs,kmax=kmax)
        else:
            self.sPzk = self.P_lin(self.ks_sigma2,self.zs)
        ks = self.ks_sigma2[None,None,:]
        R = self.R_of_m(ms)[None,:,None]
        W2 = Wkr(ks,R,self.p['Wkr_taylor_switch'])**2.
        Ps = self.sPzk[:,None,:]
        integrand = Ps*W2*ks**2./2./np.pi**2.
        sigma2 = simps(integrand,ks,axis=-1)
        return sigma2
        
    def init_mass_function(self,ms):
        self.ms = ms
        self.sigma2 = self.get_sigma2()
        self.nzm = self.get_nzm()
        self.bh = self.get_bh()

    def get_fsigmaz(self):
        sigma2 = self.sigma2
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            sigma = np.sqrt(sigma2)
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return A*np.sqrt(2.*a/np.pi)*(1+((sigma2/a/deltac**2.)**p))*(deltac/sigma)*np.exp(-a*deltac**2./2./sigma2)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            fnus = tinker.f_nu(nus,self.zs[:,None])
            return nus * fnus # note that f is actually nu*fnu !
        else:
            raise NotImplementedError
    
    def get_bh(self):
        sigma2 = self.sigma2
        deltac = self.p['st_deltac']
        if self.mode=="sheth-torman":
            A = self.p['st_A']
            a = self.p['st_a']
            p = self.p['st_p']
            return 1. + (1./deltac)*((a*deltac**2./sigma2)-1.) + (2.*p/deltac)/(1.+(a*deltac**2./sigma2)**p)
        elif self.mode=="tinker":
            nus = deltac/np.sqrt(sigma2)
            return tinker.bias(nus)
        else:
            raise NotImplementedError

    def concentration(self):
        ms = self.ms
        if self.conc=='duffy':
            if self.mdef == 'mean':
                A = self.p['duffy_A_mean']
                alpha = self.p['duffy_alpha_mean']
                beta = self.p['duffy_beta_mean']
            elif self.mdef == 'vir':
                A = self.p['duffy_A_vir']
                alpha = self.p['duffy_alpha_vir']
                beta = self.p['duffy_beta_vir']
            return duffy_concentration(ms[None,:],self.zs[:,None],A,alpha,beta,self.h)

        elif self.conc=='BHATTACHARYA':
            if self.mdef == 'vir':
                sigma2 = self.sigma2
                deltac = self.p['st_deltac']
                nus = deltac/np.sqrt(sigma2)
                Dzs = self.D_growth(z2a(self.zs))
                return BHATTACHARYA_concentration(ms[None,:], nus, Dzs[:,None], self.h)
        else:
            raise NotImplementedError

    def concentrationMW(self, mMW):
        zMW   = 0
        AA    = default_params['duffy_A_vir']
        alpha = default_params['duffy_alpha_vir']
        beta  = default_params['duffy_beta_vir']
        hub   = default_params['H0'] / 100.
        return AA*((hub*mMW/2e12)**alpha)*(1+zMW)**beta

    def get_nzm(self):
        sigma2 = self.sigma2
        ms = self.ms
        ln_sigma_inv = -0.5*np.log(sigma2)
        fsigmaz = self.get_fsigmaz()
        dln_sigma_dlnm = np.gradient(ln_sigma_inv,np.log(ms),axis=-1)
        ms = ms[None,:]
        return self.rho_matter_z(0) * fsigmaz * dln_sigma_dlnm / ms**2. 


    def add_battaglia_profile(self,name,family=None,param_override=None,
                              nxs=None,
                              xmax=None,ignore_existing=False):
        if not(ignore_existing): assert name not in self.uk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."
        if nxs is None: nxs = self.p['electron_density_profile_integral_numxs']
        if xmax is None: xmax = self.p['electron_density_profile_integral_xmax']

        
        # Set default parameters
        if family is None: family = self.p['battaglia_gas_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_gas_gamma'] = self.p['battaglia_gas_gamma']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        if param_override is not None:
            print(param_override)
            for key in param_override.keys():
                if key=='battaglia_gas_gamma':
                    pparams[key] = param_override[key]
                elif key in battaglia_defaults[family]:
                    pparams[key] = param_override[key]
                else:
                    #raise ValueError # param in param_override doesn't seem to be a Battaglia parameter
                    pass

        # Convert masses to m200critz
        rhocritz = self.rho_critical_z(self.zs)
        if self.mdef=='vir':
            delta_rhos1 = rhocritz*self.deltav(self.zs)
        elif self.mdef=='mean':
            delta_rhos1 = self.rho_matter_z(self.zs)*200.
        rvirs = self.rvir(self.ms[None,:],self.zs[:,None])
        cs = self.concentration()
        delta_rhos2 = 200.*self.rho_critical_z(self.zs)
        m200critz = mdelta_from_mdelta(self.ms,cs,delta_rhos1,delta_rhos2)
        r200critz = R_from_M(m200critz,self.rho_critical_z(self.zs)[:,None],delta=200.)

        # Generate profiles
        """
        The physical profile is rho(r) = f(2r/R200)
        We rescale this to f(x), so x = r/(R200/2) = r/rgs
        So rgs = R200/2 is the equivalent of rss in the NFW profile
        """
        omb = self.p['ombh2'] / self.h**2.
        omm = self.om0
        rhofunc = lambda x: rho_gas_generic_x(x,m200critz[...,None],self.zs[:,None,None],omb,omm,rhocritz[...,None,None],
                                    gamma=pparams['battaglia_gas_gamma'],
                                    rho0_A0=pparams['rho0_A0'],
                                    rho0_alpham=pparams['rho0_alpham'],
                                    rho0_alphaz=pparams['rho0_alphaz'],
                                    alpha_A0=pparams['alpha_A0'],
                                    alpha_alpham=pparams['alpha_alpham'],
                                    alpha_alphaz=pparams['alpha_alphaz'],
                                    beta_A0=pparams['beta_A0'],
                                    beta_alpham=pparams['beta_alpham'],
                                    beta_alphaz=pparams['beta_alphaz'])

        rgs = r200critz/2.
        cgs = rvirs/rgs
        ks,ukouts = generic_profile_fft(rhofunc,cgs,rgs[...,None],self.zs,self.ks,xmax,nxs)
        self.uk_profiles[name] = ukouts.copy()

    def add_battaglia_pres_profile(self,name,family=None,param_override=None,
                              nxs=None,
                              xmax=None,ignore_existing=False):
        if not(ignore_existing): assert name not in self.pk_profiles.keys(), "Profile name already exists."
        assert name!='nfw', "Name nfw is reserved."
        if nxs is None: nxs = self.p['electron_pressure_profile_integral_numxs']
        if xmax is None: xmax = self.p['electron_pressure_profile_integral_xmax']
        
        # Set default parameters
        if family is None: family = self.p['battaglia_pres_family'] # AGN or SH?
        pparams = {}
        pparams['battaglia_pres_gamma'] = self.p['battaglia_pres_gamma']
        pparams['battaglia_pres_alpha'] = self.p['battaglia_pres_alpha']
        pparams.update(battaglia_defaults[family])

        # Update with overrides
        if param_override is not None:
            for key in param_override.keys():
                if key in ['battaglia_pres_gamma','battaglia_pres_alpha']:
                    pparams[key] = param_override[key]
                elif key in battaglia_defaults[family]:
                    pparams[key] = param_override[key]
                else:
                    #raise ValueError # param in param_override doesn't seem to be a Battaglia parameter
                    pass

        # Convert masses to m200critz
        rhocritz = self.rho_critical_z(self.zs)
        if self.mdef=='vir':
            delta_rhos1 = rhocritz*self.deltav(self.zs)
        elif self.mdef=='mean':
            delta_rhos1 = self.rho_matter_z(self.zs)*200.
        rvirs = self.rvir(self.ms[None,:],self.zs[:,None])
        cs = self.concentration()
        delta_rhos2 = 200.*self.rho_critical_z(self.zs)
        m200critz = mdelta_from_mdelta(self.ms,cs,delta_rhos1,delta_rhos2)
        r200critz = R_from_M(m200critz,self.rho_critical_z(self.zs)[:,None],delta=200.)

        # Generate profiles
        """
        The physical profile is rho(r) = f(2r/R200)
        We rescale this to f(x), so x = r/(R200/2) = r/rgs
        So rgs = R200/2 is the equivalent of rss in the NFW profile
        """
        omb = self.p['ombh2'] / self.h**2.
        omm = self.om0
        presFunc = lambda x: P_e_generic_x(x,m200critz[...,None],r200critz[...,None],self.zs[:,None,None],omb,omm,rhocritz[...,None,None],
                                    alpha=pparams['battaglia_pres_alpha'],
                                    gamma=pparams['battaglia_pres_gamma'],
                                    P0_A0=pparams['P0_A0'],
                                    P0_alpham=pparams['P0_alpham'],
                                    P0_alphaz=pparams['P0_alphaz'],
                                    xc_A0=pparams['xc_A0'],
                                    xc_alpham=pparams['xc_alpham'],
                                    xc_alphaz=pparams['xc_alphaz'],
                                    beta_A0=pparams['beta_A0'],
                                    beta_alpham=pparams['beta_alpham'],
                                    beta_alphaz=pparams['beta_alphaz'])

        rgs = r200critz
        cgs = rvirs/rgs
        sigmaT=constants.physical_constants['Thomson cross section'][0] # units m^2
        mElect=constants.physical_constants['electron mass'][0] / default_params['mSun']# units kg
        ks,pkouts = generic_profile_fft(presFunc,cgs,rgs[...,None],self.zs,self.ks,xmax,nxs,do_mass_norm=False)
        self.pk_profiles[name] = pkouts.copy()*4*np.pi*(sigmaT/(mElect*constants.c**2))*(r200critz**3*((1+self.zs)**2/self.h_of_z(self.zs))[...,None])[...,None]


    def add_nfw_profile(self,name,numeric=False,
                        nxs=None,
                        xmax=None,ignore_existing=False):

        """
        xmax should be thought of in "concentration units", i.e.,
        for a cluster with concentration 3., xmax of 100 is probably overkill
        since the integrals are zero for x>3. However, since we are doing
        a single FFT over all profiles, we need to choose a single xmax.
        xmax of 100 is very safe for m~1e9 msolar, but xmax of 200-300
        will be needed down to m~1e2 msolar.
        nxs is the number of samples from 0 to xmax of rho_nfw(x). Might need
        to be increased from default if xmax is increased and/or going down
        to lower halo masses.
        xmax decides accuracy on large scales
        nxs decides accuracy on small scales
        """
        if not(ignore_existing): assert name not in self.uk_profiles.keys(), "Profile name already exists."
        if nxs is None: nxs = self.p['nfw_integral_numxs']
        if xmax is None: xmax = self.p['nfw_integral_xmax']
        cs = self.concentration()
        ms = self.ms
        rvirs = self.rvir(ms[None,:],self.zs[:,None])
        rss = (rvirs/cs)[...,None]
        if numeric:
            ks,ukouts = generic_profile_fft(lambda x: rho_nfw_x(x,rhoscale=1),cs,rss,self.zs,self.ks,xmax,nxs)
            self.uk_profiles[name] = ukouts.copy()
        else:
            cs = cs[...,None]
            mc = np.log(1+cs)-cs/(1.+cs)
            x = self.ks[None,None]*rss *(1+self.zs[:,None,None])# !!!!
            Si, Ci = scipy.special.sici(x)
            Sic, Cic = scipy.special.sici((1.+cs)*x)
            ukouts = (np.sin(x)*(Sic-Si) - np.sin(cs*x)/((1+cs)*x) + np.cos(x)*(Cic-Ci))/mc
            self.uk_profiles[name] = ukouts.copy()

        return self.ks, ukouts

    def add_nfw_profile_new(self, lamb, name):
        ms200, rs200, cs200 = self.mrc200()

        cs = lamb * cs200[...,None]

        mc = np.log(1+cs) - cs/(1.+cs)

        rss = (rs200/cs200)[...,None]
        q = self.ks[None,None,:] * rss *(1+self.zs[:,None,None])# !!!!

        Si, Ci = scipy.special.sici(q)
        Sic, Cic = scipy.special.sici((1.+cs)*q)

        ukouts = (np.sin(q)*(Sic-Si) - np.sin(cs*q)/((1+cs)*q) + np.cos(q)*(Cic-Ci)) / mc
        self.uk_profiles[name] = ukouts.copy()

        return self.ks, ukouts

    def add_hod(self, name, corr="max",
                satellite_profile_name='nfw',
                central_profile_name=None,
                ignore_existing=False,
                param_override=None):
        """
        Specify an HOD.
        This requires either a stellar mass threshold mthresh (nz,)
        or a number density ngal (nz,) from which mthresh is identified iteratively.
        You can either specify a corr="max" maximally correlated central-satellite 
        model or a corr="min" minimally correlated model.
        Miscentering could be included through central_profile_name (default uk=1 for default name of None).
        """
        if not(ignore_existing): 
            assert name not in self.uk_profiles.keys(), \
                "HOD name already used by profile."
        assert satellite_profile_name in self.uk_profiles.keys(), \
            "No matter profile by that name exists."
        if central_profile_name is not None:
            assert central_profile_name in self.uk_profiles.keys(), \
                "No matter profile by that name exists."
        if not(ignore_existing): 
            assert name not in self.hods.keys(), "HOD with that name already exists."

        if self.unWISE == 'blue':
            hod_params = blue_dicts[self.chooseHOD]
        elif self.unWISE == 'green':
            hod_params = green_dicts[self.chooseHOD]

        self.hods[name] = {}

        mthresh = hod_params['hod_Mmin_cendivh']/self.h

        Ncs = avg_Nc(logmhalo = np.log10(self.ms),
                     logm_minHOD = np.log10(mthresh),
                     siglogM = hod_params['hod_siglogM'])

        Nss = avg_Ns(self.ms, Nc = Ncs,
                     alphasat = hod_params['hod_alphasat'],
                     m1pr = hod_params['hod_M1pr_satdivh']/self.h)

        # added corrs by hand according to assmptions in Hill
        NsNsm1 = Nss**2.
        NcNs = Nss

        self.hods[name]['hod_lambda'] = hod_params['hod_lambda']
        self.hods[name]['Nc'] = Ncs
        self.hods[name]['Ns'] = Nss
        self.hods[name]['NsNsm1'] = NsNsm1
        self.hods[name]['NcNs'] = NcNs
        self.hods[name]['ngal'] = self.get_ngal(Ncs,Nss)
        self.hods[name]['bg'] = self.get_bg(Ncs,Nss,self.hods[name]['ngal'])
        self.hods[name]['satellite_profile'] = satellite_profile_name
        self.hods[name]['central_profile'] = central_profile_name

    def get_ngal(self,Nc,Ns):
        integrand = self.nzm * (Nc+Ns)[None,:]
        return np.trapz(integrand,self.ms,axis=-1)

    def get_bg(self,Nc,Ns,ngal):
        integrand = self.nzm * (Nc+Ns)[None,:] * self.bh
        return np.trapz(integrand,self.ms,axis=-1)/ngal

    def _get_hod_common(self,name):
        hod = self.hods[name]
        cname = hod['central_profile']
        sname = hod['satellite_profile']
        uc = 1.#if cname is None else self.uk_profiles[cname]
        us = self.uk_profiles[sname]
        return hod,uc,us

    def _get_hod_square(self,name):
        hod,uc,us = self._get_hod_common(name)
        return (2.*uc*us*hod['Ns'][...,None] + (hod['Ns']**2.)[...,None] * us**2.) / (hod['ngal'][...,None,None]**2.)

    def _get_hod(self,name,lowklim=False):
        hod,uc,us = self._get_hod_common(name)
        if lowklim:
            uc = 1
            us = 1
        return (uc*hod['Nc'][...,None] + us*hod['Ns'][...,None]) / hod['ngal'][...,None,None]

    def _get_matter(self,name,lowklim=False):
        ms = self.ms[...,None]
        uk = self.uk_profiles[name]
        if lowklim: uk = 1
        return ms*uk/self.rho_matter_z(0)

    def _get_pressure(self,name,lowklim=False):
        pk = self.pk_profiles[name].copy()
        if lowklim: pk[:,:,:] = pk[:,:,0][...,None]
        return pk

    def get_power(self,name,name2=None,verbose=True,b1=None,b2=None):
        if name2 is None: name2 = name
        return self.get_power_1halo(name,name2) + self.get_power_2halo(name,name2,verbose,b1,b2)
    
    def get_power_1halo(self,name="nfw",name2=None):
        name2 = name if name2 is None else name2
        ms = self.ms[...,None]
        mnames = self.uk_profiles.keys()
        hnames = self.hods.keys()
        pnames =self.pk_profiles.keys()
        if (name in hnames) and (name2 in hnames): 
            square_term = self._get_hod_square(name)
        elif (name in pnames) and (name2 in pnames): 
            square_term = self._get_pressure(name)**2
        else:
            square_term=1.
            for nm in [name,name2]:
                if nm in hnames:
                    square_term *= self._get_hod(nm)
                elif nm in mnames:
                    square_term *= self._get_matter(nm)
                elif nm in pnames:
                    square_term *= self._get_pressure(nm)
                else: raise ValueError
        
        integrand = self.nzm[...,None] * square_term
        return np.trapz(integrand,ms,axis=-2)*(1-np.exp(-(self.ks/self.p['kstar_damping'])**2.))
    
    def get_power_2halo(self,name="nfw",name2=None,verbose=False,b1_in=None,b2_in=None):
        name2 = name if name2 is None else name2
        
        def _2haloint(iterm):
            integrand = self.nzm[...,None] * iterm * self.bh[...,None]
            integral = np.trapz(integrand,ms,axis=-2)
            return integral

        def _get_term(iname):
            if iname in self.uk_profiles.keys():
                rterm1 = self._get_matter(iname)
                rterm01 = self._get_matter(iname,lowklim=True)
                b = 1
            elif iname in self.pk_profiles.keys():
                rterm1 = self._get_pressure(iname)
                rterm01 = self._get_pressure(iname,lowklim=True)
                print ('Check the consistency relation for tSZ')
                b = rterm01 =0
            elif iname in self.hods.keys():
                rterm1 = self._get_hod(iname)
                rterm01 = self._get_hod(iname,lowklim=True)
                b = self.get_bg(self.hods[iname]['Nc'],self.hods[iname]['Ns'],self.hods[iname]['ngal'])[:,None]
            else: raise ValueError
            return rterm1,rterm01,b
            
        ms = self.ms[...,None]


        iterm1,iterm01,b1 = _get_term(name)
        iterm2,iterm02,b2 = _get_term(name2)
        if b1_in is not None:
            b1 = b1_in.reshape((b1_in.shape[0],1))
        if b2_in is not None:
            b2 = b2_in.reshape((b1_in.shape[0],1))

        integral = _2haloint(iterm1)
        integral2 = _2haloint(iterm2)
            
        # consistency relation : Correct for part that's missing from low-mass halos to get P(k->0) = b1*b2*Plinear
        consistency1 = _2haloint(iterm01)
        consistency2 = _2haloint(iterm02)
        if verbose:
            print("Two-halo consistency1: " , consistency1,integral)
            print("Two-halo consistency2: " , consistency2,integral2)
        return self.Pzk * (integral+b1-consistency1)*(integral2+b2-consistency2)

    def sigma_1h_profiles(self,thetas,Ms,concs,sig_theta=None,delta=200,rho='mean',rho_at_z=True):
        import clusterlensing as cl
        zs = self.zs
        Ms = np.asarray(Ms)
        concs = np.asarray(concs)
        chis = self.angular_diameter_distance(zs)
        rbins = chis * thetas
        offsets = chis * sig_theta if sig_theta is not None else None
        if rho=='critical': rhofunc = self.rho_critical_z 
        elif rho=='mean': rhofunc = self.rho_matter_z
        rhoz = zs if rho_at_z else zs * 0
        Rdeltas = R_from_M(Ms,rhofunc(rhoz),delta=delta)
        rs = Rdeltas / concs
        rhocrits = self.rho_critical_z(zs)
        delta_c =  Ms / 4 / np.pi / rs**3 / rhocrits / Fcon(concs)
        smd = cl.nfw.SurfaceMassDensity(rs, delta_c, rhocrits,rbins=rbins,offsets=offsets)
        sigma = smd.sigma_nfw()
        return sigma

    def kappa_1h_profiles(self,thetas,Ms,concs,zsource,sig_theta=None,delta=200,rho='mean',rho_at_z=True):
        sigma = self.sigma_1h_profiles(thetas,Ms,concs,sig_theta=sig_theta,delta=delta,rho=rho,rho_at_z=rho_at_z)
        sigmac = self.sigma_crit(self.zs,zsource)
        return sigma / sigmac

    def kappa_2h_profiles(self,thetas,Ms,zsource,delta=200,rho='mean',rho_at_z=True,lmin=100,lmax=10000,verbose=True):
        from scipy.special import j0
        zlens = self.zs
        sigmac = self.sigma_crit(zlens,zsource)
        rhomz = self.rho_matter_z(zlens)
        chis = self.comoving_radial_distance(zlens)
        DAz = self.results.angular_diameter_distance(zlens)
        ells = self.ks*chis
        sel = np.logical_and(ells>lmin,ells<lmax)
        ells = ells[sel]
        #Array indexing is as follows:
        #[z,M,k/r]
        Ps = self.Pzk[:,sel]
        bhs = []
        for i in range(zlens.shape[0]): # vectorize this
            bhs.append( interp1d(self.ms,self.bh[i])(Ms))
        bhs = np.asarray(bhs)
        if verbose:
            print("bias ",bhs)
            print("sigmacr ", sigmac)
        ints = []
        for theta in thetas: # vectorize
            integrand = rhomz * bhs * Ps / (1+zlens)**3. / sigmac / DAz**2 * j0(ells*theta) * ells / 2./ np.pi
            ints.append( np.trapz(integrand,ells) )
        return np.asarray(ints)

"""
Mass function
"""
def R_from_M(M,rho,delta): return (3.*M/4./np.pi/delta/rho)**(1./3.) 

"""
HOD functions from Matt Johnson and Moritz Munchmeyer (modified)
"""
    
def Mstellar_halo(z,log10mhalo):
    # Function to compute the stellar mass Mstellar from a halo mass mv at redshift z.
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    # FIXME: can the for loop be removed?
    # FIXME: is the zero indexing safe?

    log10mstar = np.linspace(-18,18,4000)[None,:]
    mh = Mhalo_stellar(z,log10mstar)
    mstar = np.zeros((z.shape[0],log10mhalo.shape[-1]))
    for i in range(z.size):
        mstar[i] = np.interp(log10mhalo[0],mh[i],log10mstar[0])
    return mstar

def Mhalo_stellar_core(log10mstellar,a,Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a):
    log10M1 = M1 + M1a*(a-1)
    log10Mstar0 = Mstar00 + Mstara*(a-1)
    beta = beta0 + beta_a*(a-1)
    gamma = gamma0 + gamma_a*(a-1)
    delta = delta0 + delta_a*(a-1)
    log10mstar = log10mstellar
    log10mh = -0.5 + log10M1 + beta*(log10mstar-log10Mstar0) + 10**(delta*(log10mstar-log10Mstar0))/(1.+ 10**(-gamma*(log10mstar-log10Mstar0)))
    return log10mh

def Mhalo_stellar(z,log10mstellar):
    # Function to compute halo mass as a function of the stellar mass. arxiv 1001.0015 Table 2
    # z = list of redshifts
    # log10mhalo = log of the halo mass
    
    output = np.zeros((z.size,log10mstellar.shape[-1]))
    
    a = 1./(1+z)
    log10mstellar = log10mstellar + z*0
    
    Mstar00=10.72
    Mstara=0.55
    M1=12.35
    M1a=0.28
    beta0=0.44
    beta_a=0.18
    gamma0=1.56
    gamma_a=2.51
    delta0=0.57
    delta_a=0.17
    
    sel1 = np.where(z.reshape(-1)<=0.8)
    output[sel1] = Mhalo_stellar_core(log10mstellar[sel1],a[sel1],Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a)
    
    Mstar00=11.09
    Mstara=0.56
    M1=12.27
    M1a=-0.84
    beta0=0.65
    beta_a=0.31
    gamma0=1.12
    gamma_a=-0.53
    delta0=0.56
    delta_a=-0.12

    sel1 = np.where(z.reshape(-1)>0.8)
    output[sel1] = Mhalo_stellar_core(log10mstellar[sel1],a[sel1],Mstar00,Mstara,M1,M1a,beta0,beta_a,gamma0,gamma_a,delta0,delta_a)
    return output

def avg_Nc(logmhalo, logm_minHOD, siglogM):
    """<Nc(m)>"""
    num = logmhalo - logm_minHOD
    return 0.5*(1. + erf(num/siglogM))

def avg_Ns(ms, Nc, alphasat, m1pr):
    m0 = 0.
    num = ms - m0
    return Nc * (num / m1pr)**alphasat


"""
Profiles
"""

def Fcon(c): return (np.log(1.+c) - (c/(1.+c)))

def rhoscale_nfw(mdelta,rdelta,cdelta):
    rs = rdelta/cdelta
    V = 4.*np.pi * rs**3.
    return mdelta / V / Fcon(cdelta)

def rho_nfw_x(x,rhoscale): return rhoscale/x/(1.+x)**2.

def rho_nfw(r,rhoscale,rs): return rho_nfw_x(r/rs,rhoscale)

def mdelta_from_mdelta(M1,C1,delta_rhos1,delta_rhos2,vectorized=True):
    """
    Fast/vectorized mass definition conversion

    Converts M1(m) to M2(z,m).
    Needs concentrations C1(z,m),
    cosmic densities delta_rhos1(z), e.g. delta_rhos1(z) = Delta_vir(z)*rhoc(z)
    cosmic densities delta_rhos2(z), e.g. delta_rhos2(z) = 200*rhom(z)

    The vectorized version is several orders of magnitude faster.
    """
    if vectorized:
        M1 = M1[None,:]+C1*0.
        M2outs =  mdelta_from_mdelta_unvectorized(M1.copy(),C1,delta_rhos1[:,None],delta_rhos2[:,None])
    else:
        M2outs = np.zeros(C1.shape)
        for i in range(C1.shape[0]):
            for j in range(C1.shape[1]):
                M2outs[i,j] = mdelta_from_mdelta_unvectorized(M1[j],C1[i,j],delta_rhos1[i],delta_rhos2[i])
    return M2outs

    
def mdelta_from_mdelta_unvectorized(M1,C1,delta_rhos1,delta_rhos2):
    """
    Implements mdelta_from_mdelta.
    The logMass is necessary for numerical stability.
    I thought I calculated the right derivative, but using it leads to wrong
    answers, so the secant method is used instead of Newton's method.
    In principle, both the first and second derivatives can be calculated
    analytically.

    The conversion is done by assuming NFW and solving the equation
    M1 F1 - M2 F2 = 0
    where F(conc) = 1 / (log(1+c) - c/(1+c))
    This equation is obtained when rhoscale is equated between the two mass
    definitions, where rhoscale = F(c) * m /(4pi rs**3) is the amplitude
    of the NFW profile. The scale radii are also the same. Equating
    the scale radii also provides
    C2 = ((M2/M1) * (delta_rhos1/delta_rhos2) * (rho1/rho2)) ** (1/3) C1
    which reduces the system to one unknown M2.
    """
    C2 = lambda logM2: C1*((np.exp(logM2-np.log(M1)))*(delta_rhos1/delta_rhos2))**(1./3.)
    F2 = lambda logM2: 1./Fcon(C2(logM2))
    F1 = 1./Fcon(C1)
    # the function whose roots to find
    func = lambda logM2: M1*F1 - np.exp(logM2)*F2(logM2)
    from scipy.optimize import newton
    # its analytical derivative
    #jaco = lambda logM2: -F2(logM2) + (C2(logM2)/(1.+C2(logM2)))**2. * C2(logM2)/3. * F2(logM2)**2.
    M2outs = newton(func,np.log(M1))#,fprime=jaco) # FIXME: jacobian doesn't work
    return np.exp(M2outs)

def battaglia_gas_fit(m200critz,z,A0x,alphamx,alphazx):
    # Any factors of h in M?
    return A0x * (m200critz/1.e14)**alphamx * (1.+z)**alphazx
    
def rho_gas(r,m200critz,z,omb,omm,rhocritz,
            gamma=default_params['battaglia_gas_gamma'],
            profile="AGN"):
    return rho_gas_generic(r,m200critz,z,omb,omm,rhocritz,
                           gamma=gamma,
                           rho0_A0=battaglia_defaults[profile]['rho0_A0'],
                           rho0_alpham=battaglia_defaults[profile]['rho0_alpham'],
                           rho0_alphaz=battaglia_defaults[profile]['rho0_alphaz'],
                           alpha_A0=battaglia_defaults[profile]['alpha_A0'],
                           alpha_alpham=battaglia_defaults[profile]['alpha_alpham'],
                           alpha_alphaz=battaglia_defaults[profile]['alpha_alphaz'],
                           beta_A0=battaglia_defaults[profile]['beta_A0'],
                           beta_alpham=battaglia_defaults[profile]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[profile]['beta_alphaz'])

def rho_gas_generic(r,m200critz,z,omb,omm,rhocritz,
                    gamma=default_params['battaglia_gas_gamma'],
                    rho0_A0=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_A0'],
                    rho0_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alpham'],
                    rho0_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alphaz'],
                    alpha_A0=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_A0'],
                    alpha_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alpham'],
                    alpha_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alphaz'],
                    beta_A0=battaglia_defaults[default_params['battaglia_gas_family']]['beta_A0'],
                    beta_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alpham'],
                    beta_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alphaz'],
):
    """
    AGN and SH Battaglia 2016 profiles
    r: physical distance
    m200critz: M200_critical_z

    """
    R200 = R_from_M(m200critz,rhocritz,delta=200)
    x = 2*r/R200
    return rho_gas_generic_x(x,m200critz,z,omb,omm,rhocritz,gamma,
                             rho0_A0,rho0_alpham,rho0_alphaz,
                             alpha_A0,alpha_alpham,alpha_alphaz,
                             beta_A0,beta_alpham,beta_alphaz)

def rho_gas_generic_x(x,m200critz,z,omb,omm,rhocritz,
                    gamma=default_params['battaglia_gas_gamma'],
                    rho0_A0=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_A0'],
                    rho0_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alpham'],
                    rho0_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['rho0_alphaz'],
                    alpha_A0=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_A0'],
                    alpha_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alpham'],
                    alpha_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['alpha_alphaz'],
                    beta_A0=battaglia_defaults[default_params['battaglia_gas_family']]['beta_A0'],
                    beta_alpham=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alpham'],
                    beta_alphaz=battaglia_defaults[default_params['battaglia_gas_family']]['beta_alphaz'],
):
    rho0 = battaglia_gas_fit(m200critz,z,rho0_A0,rho0_alpham,rho0_alphaz)
    alpha = battaglia_gas_fit(m200critz,z,alpha_A0,alpha_alpham,alpha_alphaz)
    beta = battaglia_gas_fit(m200critz,z,beta_A0,beta_alpham,beta_alphaz)
    # Note the sign difference in the second gamma. Battaglia 2016 had a typo here.
    return (omb/omm) * rhocritz * rho0 * (x**gamma) * (1.+x**alpha)**(-(beta+gamma)/alpha)


   
def P_e(r,m200critz,z,omb,omm,rhocritz,
            alpha=default_params['battaglia_pres_alpha'],
            gamma=default_params['battaglia_pres_gamma'],
            profile="pres"):
    return P_e_generic(r,m200critz,z,omb,omm,rhocritz,
                           alpha=alpha,
                           gamma=gamma,
                           P0_A0=battaglia_defaults[profile]['P0_A0'],
                           P0_alpham=battaglia_defaults[profile]['P0_alpham'],
                           P0_alphaz=battaglia_defaults[profile]['P0_alphaz'],
                           xc_A0=battaglia_defaults[profile]['xc_A0'],
                           xc_alpham=battaglia_defaults[profile]['xc_alpham'],
                           xc_alphaz=battaglia_defaults[profile]['xc_alphaz'],
                           beta_A0=battaglia_defaults[profile]['beta_A0'],
                           beta_alpham=battaglia_defaults[profile]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[profile]['beta_alphaz'])

def P_e_generic(r,m200critz,z,omb,omm,rhocritz,
                        alpha=default_params['battaglia_pres_alpha'],
                        gamma=default_params['battaglia_pres_gamma'],
                           P0_A0=battaglia_defaults[default_params['battaglia_pres_family']]['P0_A0'],
                           P0_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['P0_alpham'],
                           P0_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['P0_alphaz'],
                           xc_A0=battaglia_defaults[default_params['battaglia_pres_family']]['xc_A0'],
                           xc_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['xc_alpham'],
                           xc_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['xc_alphaz'],
                           beta_A0=battaglia_defaults[default_params['battaglia_pres_family']]['beta_A0'],
                           beta_alpham=battaglia_defaults[default_params['battaglia_pres_family']]['beta_alpham'],
                           beta_alphaz=battaglia_defaults[default_params['battaglia_pres_family']]['beta_alphaz']):
    """
    AGN and SH Battaglia 2016 profiles
    r: physical distance
    m200critz: M200_critical_z

    """
    R200 = R_from_M(m200critz,rhocritz,delta=200)
    x = r/R200
    return P_e_generic_x(x,m200critz,R200,z,omb,omm,rhocritz,alpha,gamma,
                             P0_A0,P0_alpham,P0_alphaz,
                             xc_A0,xc_alpham,xc_alphaz,
                             beta_A0,beta_alpham,beta_alphaz)

def P_e_generic_x(x,m200critz,R200critz,z,omb,omm,rhocritz,
                  alpha=default_params['battaglia_pres_alpha'],
                  gamma=default_params['battaglia_pres_gamma'],
                  P0_A0=battaglia_defaults['pres']['P0_A0'],
                  P0_alpham=battaglia_defaults['pres']['P0_alpham'],
                  P0_alphaz=battaglia_defaults['pres']['P0_alphaz'],
                  xc_A0=battaglia_defaults['pres']['xc_A0'],
                  xc_alpham=battaglia_defaults['pres']['xc_alpham'],
                  xc_alphaz=battaglia_defaults['pres']['xc_alphaz'],
                  beta_A0=battaglia_defaults['pres']['beta_A0'],
                  beta_alpham=battaglia_defaults['pres']['beta_alpham'],
                  beta_alphaz=battaglia_defaults['pres']['beta_alphaz']):
    P0 = battaglia_gas_fit(m200critz,z,P0_A0,P0_alpham,P0_alphaz)
    xc = battaglia_gas_fit(m200critz,z,xc_A0,xc_alpham,xc_alphaz)
    beta = battaglia_gas_fit(m200critz,z,beta_A0,beta_alpham,beta_alphaz)
    # to convert to p_e
    XH=.76
    eFrac=2.0*(XH+1.0)/(5.0*XH+3.0)
    # print (gamma,alpha,beta[0,0],xc[0,0],P0[0,0])
    # print (gamma,alpha,beta[0,50],xc[0,50],P0[0,50],(P0 *(x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta))[0,50,-350])
    G_newt = constants.G/(default_params['parsec']*1e6)**3*default_params['mSun']
    return eFrac*(omb/omm)*200*m200critz*G_newt* rhocritz/(2*R200critz) * P0 * (x/xc)**gamma * (1.+(x/xc)**alpha)**(-beta)

def a2z(a): return 1./a - 1.
def z2a(z): return 1./(1. + z)


# Original paper best fit params
# HOD unWISE Blue
dict_blue0 = {'hod_alphasat': 1.304,
            'hod_siglogM': 0.687,
            'hod_lambda' : 1.087,
            'hod_M1pr_satdivh' : 5.027e12,
            'hod_Mmin_cendivh' : 6.251e11}

# HOD unWISE green
dict_green0 = {'hod_alphasat': 1.302,
            'hod_siglogM': 0.973,
            'hod_lambda' : 2.746,
            'hod_M1pr_satdivh' : 2.760e13,
            'hod_Mmin_cendivh' : 1.342e13}

# New paper best fit params
# HOD unWISE Blue
new_dict_blue = {'hod_alphasat': 1.06,
                'hod_siglogM': 0.02,
                'hod_lambda' : 1.80,
                'hod_M1pr_satdivh' : 4.074e12,
                'hod_Mmin_cendivh' : 4.8978e11}

# HOD unWISE green
new_dict_green = {'hod_alphasat': 1.14,
                'hod_siglogM': 0.03,
                'hod_lambda' : 0.94,
                'hod_M1pr_satdivh' : 1.47911e13,
                'hod_Mmin_cendivh' : 1.69824e12}

# Random HODs from the chains

dict_blue1 = {'hod_alphasat': 1.3542459 ,\
              'hod_siglogM': 0.98677494 ,\
              'hod_lambda': 1.2582939   ,\
              'hod_M1pr_satdivh': 12906599000000.0,\
              'hod_Mmin_cendivh': 2953068500000.0}
dict_blue2 = {'hod_alphasat': 1.4253546 ,\
              'hod_siglogM': 0.81630793 ,\
              'hod_lambda': 1.3982947   ,\
              'hod_M1pr_satdivh': 20948455000000.0,\
              'hod_Mmin_cendivh': 2607282000000.0}
dict_blue3 = {'hod_alphasat': 1.2793517 ,\
              'hod_siglogM': 1.0454666  , \
              'hod_lambda': 1.1190134   ,\
              'hod_M1pr_satdivh': 7617692000000.0,\
              'hod_Mmin_cendivh': 2504466700000.0}
dict_blue4 = {'hod_alphasat': 1.5189991 ,\
              'hod_siglogM': 0.83185432 ,\
              'hod_lambda': 0.94837449  ,\
              'hod_M1pr_satdivh': 17794764000000.0,\
              'hod_Mmin_cendivh': 1969916700000.0}
dict_blue5 = {'hod_alphasat': 1.3599408 ,\
              'hod_siglogM': 0.66925685 ,\
              'hod_lambda': 1.1812971   ,\
              'hod_M1pr_satdivh': 9746309400000.0,\
              'hod_Mmin_cendivh': 1026943100000.0}
dict_blue6 = {'hod_alphasat': 1.2959718 ,\
              'hod_siglogM': 0.90888251 ,\
              'hod_lambda': 1.2190117   ,\
              'hod_M1pr_satdivh': 7002756800000.0,\
              'hod_Mmin_cendivh': 1445789600000.0}
dict_blue7 = {'hod_alphasat': 1.2038421 ,\
              'hod_siglogM': 1.1451285  ,\
              'hod_lambda': 1.0946323   ,\
              'hod_M1pr_satdivh': 9599372000000.0, \
              'hod_Mmin_cendivh': 5663054000000.0}
dict_blue8 = {'hod_alphasat': 1.3218181 ,\
              'hod_siglogM': 1.0815548  ,\
              'hod_lambda': 1.0145024   ,\
              'hod_M1pr_satdivh': 11651399000000.0,\
              'hod_Mmin_cendivh': 4035599600000.0}
dict_blue9 = {'hod_alphasat': 1.3267543 ,\
              'hod_siglogM': 1.0773637  ,\
              'hod_lambda': 1.266267    ,\
              'hod_M1pr_satdivh': 12653725000000.0,\
              'hod_Mmin_cendivh': 4096273500000.0}
dict_blue10= {'hod_alphasat': 1.4473675 ,\
              'hod_siglogM': 0.85445106 ,\
              'hod_lambda': 1.158995    ,\
              'hod_M1pr_satdivh': 17805114000000.0,\
              'hod_Mmin_cendivh': 2370330500000.0}
dict_blue11= {'hod_alphasat': 1.4849744 ,\
              'hod_siglogM': 0.29637227 ,\
              'hod_lambda': 0.88264357  ,\
              'hod_M1pr_satdivh': 11679579000000.0,\
              'hod_Mmin_cendivh': 558891300000.0}
dict_blue12= {'hod_alphasat': 1.3457663 ,\
              'hod_siglogM': 0.9875303  ,\
              'hod_lambda': 1.2721422   ,\
              'hod_M1pr_satdivh': 22511823000000.0,\
              'hod_Mmin_cendivh': 5407451900000.0}
dict_blue13= {'hod_alphasat': 1.3478516 ,\
              'hod_siglogM': 1.0250143  ,\
              'hod_lambda': 1.3256746   ,\
              'hod_M1pr_satdivh': 24542836000000.0,\
              'hod_Mmin_cendivh': 6505173200000.0}
dict_blue14= {'hod_alphasat': 1.3834033 ,\
              'hod_siglogM': 0.41235214 ,\
              'hod_lambda': 1.0712581   ,\
              'hod_M1pr_satdivh': 6139530300000.0,\
              'hod_Mmin_cendivh': 391815100000.0}
dict_blue15= {'hod_alphasat': 1.317003  ,\
              'hod_siglogM': 1.0885908  ,\
              'hod_lambda': 1.0964565   ,\
              'hod_M1pr_satdivh': 11700566000000.0,\
              'hod_Mmin_cendivh': 4029518900000.0}
dict_blue16= {'hod_alphasat': 1.2765424 ,\
              'hod_siglogM': 0.72878949 ,\
              'hod_lambda': 1.2253719   ,\
              'hod_M1pr_satdivh': 5454359600000.0,\
              'hod_Mmin_cendivh': 794261010000.0}
dict_blue17= {'hod_alphasat': 1.2704002 ,\
              'hod_siglogM': 0.092796437,\
              'hod_lambda': 1.4252134   ,\
              'hod_M1pr_satdivh': 3814942700000.0,\
              'hod_Mmin_cendivh': 270773660000.0}
dict_blue18= {'hod_alphasat': 1.2624443 ,\
              'hod_siglogM': 1.0914074  ,\
              'hod_lambda': 1.087492    ,\
              'hod_M1pr_satdivh': 8108832100000.0,\
              'hod_Mmin_cendivh': 3268434600000.0}
dict_blue19= {'hod_alphasat': 1.2979475 ,\
              'hod_siglogM': 0.73107902 ,\
              'hod_lambda': 1.5710729   ,\
              'hod_M1pr_satdivh': 8600338100000.0,\
              'hod_Mmin_cendivh': 1138594500000.0}
dict_blue20= {'hod_alphasat': 1.2908418 ,\
              'hod_siglogM': 0.014744779,\
              'hod_lambda': 1.3435444   ,\
              'hod_M1pr_satdivh': 4388065900000.0,\
              'hod_Mmin_cendivh': 294505120000.0}

dict_green1 = {'hod_alphasat': 1.1199285 ,\
               'hod_siglogM': 0.2559485  ,\
               'hod_lambda': 2.3204953   ,\
               'hod_M1pr_satdivh': 739609870000.0,\
               'hod_Mmin_cendivh': 386339490000.0}
dict_green2 = {'hod_alphasat': 1.3093428 ,\
               'hod_siglogM': 0.6902569  ,\
               'hod_lambda': 2.1182514   ,\
               'hod_M1pr_satdivh': 14737970000000.0,\
               'hod_Mmin_cendivh': 3452507200000.0}
dict_green3 = {'hod_alphasat': 1.144764  ,\
               'hod_siglogM': 0.11554094 ,\
               'hod_lambda': 2.5390308   ,\
               'hod_M1pr_satdivh': 3413038500000.0,\
               'hod_Mmin_cendivh': 839325340000.0}
dict_green4 = {'hod_alphasat': 1.4621949 ,\
               'hod_siglogM': 0.74991013 ,\
               'hod_lambda': 2.6610008   ,\
               'hod_M1pr_satdivh': 47286370000000.0,\
               'hod_Mmin_cendivh': 7948293600000.0}
dict_green5 = {'hod_alphasat': 1.2259404 ,\
               'hod_siglogM': 0.44803261 ,\
               'hod_lambda': 2.6475006   ,\
               'hod_M1pr_satdivh': 8234333700000.0,\
               'hod_Mmin_cendivh': 1616679900000.0}
dict_green6 = {'hod_alphasat': 1.1555298 ,\
               'hod_siglogM': 0.54551671 ,\
               'hod_lambda': 1.8841566   ,\
               'hod_M1pr_satdivh': 1071075400000.0,\
               'hod_Mmin_cendivh': 528063640000.0}
dict_green7 = {'hod_alphasat': 1.1565689 ,\
               'hod_siglogM': 0.44166128 ,\
               'hod_lambda': 2.2375547   ,\
               'hod_M1pr_satdivh': 2100496100000.0,\
               'hod_Mmin_cendivh': 741700700000.0}
dict_green8 = {'hod_alphasat': 1.1480555 ,\
               'hod_siglogM': 0.32744972 ,\
               'hod_lambda': 2.6679595   ,\
               'hod_M1pr_satdivh': 1606752300000.0,\
               'hod_Mmin_cendivh': 570039620000.0}
dict_green9 = {'hod_alphasat': 1.1696466 ,\
               'hod_siglogM': 0.13384796 ,\
               'hod_lambda': 2.058867    ,\
               'hod_M1pr_satdivh': 1948530800000.0,\
               'hod_Mmin_cendivh': 544294070000.0}
dict_green10= {'hod_alphasat': 1.2234061 ,\
               'hod_siglogM': 0.40710593 ,\
               'hod_lambda': 1.9589552   ,\
               'hod_M1pr_satdivh': 5540096300000.0,\
               'hod_Mmin_cendivh': 1186681200000.0}
dict_green11= {'hod_alphasat': 1.2795335 ,\
               'hod_siglogM': 0.85873585 ,\
               'hod_lambda': 2.7663838   ,\
               'hod_M1pr_satdivh': 19204442000000.0,\
               'hod_Mmin_cendivh': 7020248100000.0}
dict_green12= {'hod_alphasat': 1.3163661 ,\
               'hod_siglogM': 0.95931557 ,\
               'hod_lambda': 2.0507178   ,\
               'hod_M1pr_satdivh': 28228848000000.0,\
               'hod_Mmin_cendivh': 12976212000000.0}
dict_green13= {'hod_alphasat': 1.2047022 ,\
               'hod_siglogM': 0.43739616 ,\
               'hod_lambda': 2.9709142   ,\
               'hod_M1pr_satdivh': 4721159900000.0,\
               'hod_Mmin_cendivh': 1127288600000.0}
dict_green14= {'hod_alphasat': 1.3628552 ,\
               'hod_siglogM': 0.95220679 ,\
               'hod_lambda': 2.3919751   ,\
               'hod_M1pr_satdivh': 28789854000000.0,\
               'hod_Mmin_cendivh': 12039295000000.0}
dict_green15= {'hod_alphasat': 1.2751323 ,\
               'hod_siglogM': 0.54853491 ,\
               'hod_lambda': 1.9949132   ,\
               'hod_M1pr_satdivh': 9318615600000.0,\
               'hod_Mmin_cendivh': 1934864000000.0}
dict_green16= {'hod_alphasat': 1.1719873 ,\
               'hod_siglogM': 0.51853255 ,\
               'hod_lambda': 2.4484238   ,\
               'hod_M1pr_satdivh': 1864269100000.0,\
               'hod_Mmin_cendivh': 702060550000.0}
dict_green17= {'hod_alphasat': 1.3921079 ,\
               'hod_siglogM': 0.15494248 ,\
               'hod_lambda': 2.1452648   ,\
               'hod_M1pr_satdivh': 16637497000000.0,\
               'hod_Mmin_cendivh': 1507554400000.0}
dict_green18= {'hod_alphasat': 1.4143309 ,\
               'hod_siglogM': 0.3138574  ,\
               'hod_lambda': 2.3488559   ,\
               'hod_M1pr_satdivh': 14022091000000.0,\
               'hod_Mmin_cendivh': 1451449200000.0}
dict_green19= {'hod_alphasat': 1.1365154 ,\
               'hod_siglogM': 1.049071   ,\
               'hod_lambda': 2.6907772   ,\
               'hod_M1pr_satdivh': 22408731000000.0,\
               'hod_Mmin_cendivh': 19306232000000.0}
dict_green20= {'hod_alphasat': 1.5113919 ,\
               'hod_siglogM': 0.031345002,\
               'hod_lambda': 2.0062316   ,\
               'hod_M1pr_satdivh': 22168252000000.0,\
               'hod_Mmin_cendivh': 1462530200000.0}


blue_dicts = np.array([dict_blue0,\
              dict_blue1,  dict_blue2,  dict_blue3,  dict_blue4,  dict_blue5 ,\
              dict_blue6,  dict_blue7,  dict_blue8,  dict_blue9,  dict_blue10,\
              dict_blue11, dict_blue12, dict_blue13, dict_blue14, dict_blue15,\
              dict_blue16, dict_blue17, dict_blue18, dict_blue19, dict_blue20,\
              new_dict_blue])

green_dicts = np.array([dict_green0,\
               dict_green1,  dict_green2,  dict_green3,  dict_green4,  dict_green5 ,\
               dict_green6,  dict_green7,  dict_green8,  dict_green9,  dict_green10,\
               dict_green11, dict_green12, dict_green13, dict_green14, dict_green15,\
               dict_green16, dict_green17, dict_green18, dict_green19, dict_green20,\
              new_dict_green])
