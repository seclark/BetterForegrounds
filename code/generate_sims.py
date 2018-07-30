import numpy as np
import glob, pickle
import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
                  'weight' : 'normal', 'size' : 16}
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
"""
code to make simulated Q/U maps with which to test power spectrum estimation pipelines
"""

# directory in which to save simulated maps
sim_dir = '/data/jch/Planckdata/DustSims/'

# map size
Nside=2048
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
pl = hp.sphtfunc.pixwin(Nside, pol=True)
plT = pl[0][0:ellmax+1] #up to ellmax
plP = pl[1][0:ellmax+1] #up to ellmax

# generate simulated maps
Nsim = 100
for i in xrange(Nsim):
    print i
    simmap = hp.sphtfunc.synfast(Cellarr, nside=Nside, lmax=int(ellmax), pol=True, pixwin=True, new=True, verbose=False)
    # save simulated TQU maps
    hp.fitsfunc.write_map(sim_dir+'DustSim_BICEPamp_alpha-2.42_TQU_'+str(i)+'.fits', simmap, coord='G', overwrite=True)
    # # measure power spectra of simulated maps to check that pixel window is applied correctly
    # simCl = hp.sphtfunc.anafast(simmap, lmax=4000, pol=True)
    # simClEE = simCl[1] / plP**2.0 # correct for pixel window
    # simClBB = simCl[2] / plP**2.0 # correct for pixel window
    # # plot vs. theory input
    # plt.clf()
    # plt.semilogy(ells, EEDust, 'b', lw=1.5, label='EE (input)')
    # plt.semilogy(ells, simClEE, 'k', lw=1., label='EE (sim.)')
    # plt.semilogy(ells, BBDust, 'r', lw=1.5, label='BB (input)')
    # plt.semilogy(ells, simClBB, 'm', lw=1., label='BB (sim.)')
    # plt.xlim(ellmin,ellmax)
    # #plt.ylim
    # plt.xlabel(r'$\ell$')
    # plt.ylabel(r'$C_{\ell}$')
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(sim_dir+'DustSim_BICEPamp_alpha-2.42_TQU_'+str(i)+'_Clcomp.pdf')
    # # loglog
    # plt.clf()
    # plt.loglog(ells, EEDust, 'b', lw=1.5, label='EE (input)')
    # plt.loglog(ells, simClEE, 'k', lw=1., label='EE (sim.)')
    # plt.loglog(ells, BBDust, 'r', lw=1.5, label='BB (input)')
    # plt.loglog(ells, simClBB, 'm', lw=1., label='BB (sim.)')
    # plt.xlim(ellmin,ellmax)
    # #plt.ylim
    # plt.xlabel(r'$\ell$')
    # plt.ylabel(r'$C_{\ell}$')
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(sim_dir+'DustSim_BICEPamp_alpha-2.42_TQU_'+str(i)+'_Clcomploglog.pdf')
    # # ratio
    # plt.clf()
    # plt.plot(ells, simClEE/EEDust, 'k', lw=1., label='EE (sim./inp.)')
    # plt.plot(ells, simClBB/BBDust, 'm', lw=1., label='BB (sim./inp.)')
    # plt.xlim(ellmin,ellmax)
    # plt.ylim(0.95,1.05)
    # plt.xlabel(r'$\ell$')
    # plt.ylabel(r'$C_{\ell}$ ratio')
    # plt.grid()
    # plt.legend(loc='upper right')
    # plt.savefig(sim_dir+'DustSim_BICEPamp_alpha-2.42_TQU_'+str(i)+'_Clcompratio.pdf')
