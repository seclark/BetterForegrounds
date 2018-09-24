import numpy as np
import healpy as hp
"""
code to make simulated Q/U maps with which to test power spectrum estimation pipelines
"""

# directory in which to save simulated maps
sim_dir = '/data/seclark/SimMaps/'

# map size
Nside=1024
Npix = 12*Nside**2

# set up toy power spectra
ellmin = 0
ellmax = 4000
ells = np.arange(ellmin,ellmax+1)

# choose an ell at which to normalize the dust power spectrum amplitude
ellnormDust = 80.
ampDustEE = 7.93 / ellnormDust / (ellnormDust+1.0) * 2.0 * np.pi #polarized dust EE amplitude at 353 GHz in uK^2 in BICEP patch in D_ell convention; convert to C_ell
ampDustBB = 0.5*ampDustEE #approx estimate
# dust EE and BB power scaling with ell (based on Planck XXX) -- THIS IS THE SCALING OF C_ELL, not ell*(ell+1)/2pi*C_ell
alphaDust = -2.42

# construct power spectra
EEDust = ampDustEE*(ells/ellnormDust)**alphaDust 
BBDust = ampDustBB*(ells/ellnormDust)**alphaDust
# remove NaN in ell=0
EEDust[0] = 0.0
BBDust[0] = 0.0

# set TT and TE power spectra to zero, since we're not interested in these here
TTnull = np.zeros(len(ells))
TEnull = np.zeros(len(ells))
# update: polspice will not run if one of the maps is all zeros, so replace the T map with white noise
TTwhite = np.ones(len(ells))*ampDustEE

# set up tuple of C_ell as required by healpix
Cellarr = (TTwhite, EEDust, BBDust, TEnull)
# save EE and BB power spectra
np.savetxt(sim_dir+'DustEEBB_BICEPamp_alpha-2.42.txt', np.transpose(np.array([EEDust,BBDust])))

# pixel window
pixwin = True
if pixwin:
    pl = hp.sphtfunc.pixwin(Nside, pol=True)
    plT = pl[0][0:ellmax+1] #up to ellmax
    plP = pl[1][0:ellmax+1] #up to ellmax

# generate simulated maps
Nsim = 100
for i in xrange(Nsim):
    print i
    simmap = hp.sphtfunc.synfast(Cellarr, nside=Nside, lmax=int(ellmax), pol=True, pixwin=pixwin, new=True, verbose=False)
    # save simulated TQU maps
    hp.fitsfunc.write_map(sim_dir+'DustSim_BICEPamp_alpha-2.42_TQU_pixwin{}_{}.fits'.format(pixwin, i), simmap, coord='G', overwrite=True)
    
    

    