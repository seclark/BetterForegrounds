import numpy as np
import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt
#import pyfits
import healpy as hp
import subprocess
import os
### janky NaMaster import ###
try:
    import pymaster as nmt
except ImportError:
    #print "no pymaster module found, but we try again and it works hooray"
    import pymaster as nmt
### --------------------- ###

#####
FITS_end = '.fits'
TXT_end = '.txt'
PDF_end = '.pdf'
DAT_end = '.dat'
#####

#####
# mask parameters
N_side=2048
Npix = 12*N_side**2
Coords = 'G' #coordinate system
mask_dir = '/data/jch/RHT_QU/'
mask_name = 'allsky_GALFA_mask_nonzero_nchannels_edge200_hp_proj_plusbgt70_maskGal'
# apodization scale
apod_Arcmin = 60.
apod_Deg = apod_Arcmin/60.
# mask file name
mask_file = mask_dir+mask_name+FITS_end
# read in mask and apodize
# to use the pure-B formalism, need the mask to be differentiable at the boundary; the "C1" and "C2" schemes in nmt satisfy this criterion
mask = hp.read_map(mask_file, verbose=False)
apod_Type = 'C2'
#print "apodizing mask"
mask_apod = nmt.mask_apodization(mask, apod_Deg, apotype=apod_Type)
#print "done apodizing"
#####

#####
# simulation directory, maps, and theory power spectra
sim_dir = '/data/jch/Planckdata/DustSims/'
sim_name = 'DustSim_BICEPamp_alpha-2.42_TQU_' #'+str(i)+'.fits'
Nsim = 10 #actually have 100
sim_theory = 'DustEEBB_BICEPamp_alpha-2.42'
EEDust = (np.loadtxt(sim_dir+sim_theory+TXT_end))[:,0] #ClEE -- extends to ell=4000
BBDust = (np.loadtxt(sim_dir+sim_theory+TXT_end))[:,1] #ClBB -- extends to ell=4000
ellmax = 1001
ell = np.arange(int(ellmax)+1)
EEDust = EEDust[0:int(ellmax)+1]
BBDust = BBDust[0:int(ellmax)+1]
EBDust = np.zeros(ellmax+1)
BEDust = np.zeros(ellmax+1)
#####

#####
# define binning scheme
binwidth = 20
bins = nmt.NmtBin(N_side, nlb=binwidth, lmax=int(ellmax))
ell_binned = bins.get_effective_ells()
nbins = len(ell_binned)
#####

#####
# read in the zeroth sim and compute the mode-coupling matrix for non-pure and pure fields
Q, U = hp.read_map(sim_dir+sim_name+str(0)+FITS_end, field=[1, 2], verbose=False)
# non-pure
EB_npure = nmt.NmtField(mask_apod, [Q,U])
#print "mask and maps read in"
w_npure = nmt.NmtWorkspace()
#print "computing coupling matrix"
w_npure.compute_coupling_matrix(EB_npure, EB_npure, bins)
#print "coupling matrix done"
# pure
EB_pure = nmt.NmtField(mask_apod, [Q,U], purify_e = True, purify_b = True)
#print "mask and maps read in (pure)"
w_pure = nmt.NmtWorkspace()
#print "computing coupling matrix (pure)"
w_pure.compute_coupling_matrix(EB_pure, EB_pure, bins)
#print "coupling matrix done (pure)"
#####

#####
# get theory predictions with mode-coupling accounted for
ClDust_binned_npure = w_npure.decouple_cell(w_npure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
ClDust_binned_pure = w_pure.decouple_cell(w_pure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
#####

#####
# loop over sims
ClEE_arr = np.zeros((Nsim,nbins))
ClBB_arr = np.zeros((Nsim,nbins))
ClEE_pure_arr = np.zeros((Nsim,nbins))
ClBB_pure_arr = np.zeros((Nsim,nbins))
# zeroth sim is already done
# non-pure
print "0"
Cl_2x2 = w_npure.decouple_cell(nmt.compute_coupled_cell(EB_npure,EB_npure)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
ClEE_arr[0] = Cl_2x2[0]
ClBB_arr[0] = Cl_2x2[3]
# pure
Cl_2x2_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_pure,EB_pure)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
ClEE_pure_arr[0] = Cl_2x2_pure[0]
ClBB_pure_arr[0] = Cl_2x2_pure[3]
# rest of sims
for i in xrange(1,Nsim):
    print i
    Q, U = hp.read_map(sim_dir+sim_name+str(i)+FITS_end, field=[1, 2], verbose=False)
    EB_npure = nmt.NmtField(mask_apod, [Q,U])
    EB_pure = nmt.NmtField(mask_apod, [Q,U], purify_e = True, purify_b = True)
    # non-pure
    Cl_2x2 = w_npure.decouple_cell(nmt.compute_coupled_cell(EB_npure,EB_npure)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    ClEE_arr[i] = Cl_2x2[0]
    ClBB_arr[i] = Cl_2x2[3]
    # pure
    Cl_2x2_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_pure,EB_pure)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    ClEE_pure_arr[i] = Cl_2x2_pure[0]
    ClBB_pure_arr[i] = Cl_2x2_pure[3]
# avg
ClEE_mean = np.zeros(nbins)
ClBB_mean = np.zeros(nbins)
ClEE_std  = np.zeros(nbins)
ClBB_std  = np.zeros(nbins)
ClEE_pure_mean = np.zeros(nbins)
ClBB_pure_mean = np.zeros(nbins)
ClEE_pure_std  = np.zeros(nbins)
ClBB_pure_std  = np.zeros(nbins)
for b in xrange(nbins):
    ClEE_mean[b] = np.mean(ClEE_arr[:,b])
    ClBB_mean[b] = np.mean(ClBB_arr[:,b])
    ClEE_std[b]  = np.std(ClEE_arr[:,b])
    ClBB_std[b]  = np.std(ClBB_arr[:,b])
    ClEE_pure_mean[b] = np.mean(ClEE_pure_arr[:,b])
    ClBB_pure_mean[b] = np.mean(ClBB_pure_arr[:,b])
    ClEE_pure_std[b]  = np.std(ClEE_pure_arr[:,b])
    ClBB_pure_std[b]  = np.std(ClBB_pure_arr[:,b])
#####




#####
# Plot results - non-pure
plt.clf()
plt.semilogy(ell, EEDust[0:int(ellmax)+1], 'k', lw=1.5, alpha=0.7, label='input theory (EE)')
plt.semilogy(ell, BBDust[0:int(ellmax)+1], 'k', lw=1.5, alpha=0.7, ls='--', label='input theory (BB)')
plt.semilogy(ell_binned, ClDust_binned_npure[0], 'bo', label='convolved input theory (EE)')
plt.semilogy(ell_binned*1.1, ClDust_binned_npure[3], 'mo', label='convolved input theory (BB)')
plt.errorbar(ell_binned, ClEE_mean, yerr=[ClEE_std,ClEE_std], fmt='c', ecolor='c', elinewidth=1.5, capsize=3, capthick=1, marker='.', label='simulations (EE)')
plt.errorbar(ell_binned*1.1, ClBB_mean, yerr=[ClBB_std,ClBB_std], fmt='g', ecolor='g', elinewidth=1.5, capsize=3, capthick=1, marker='.', label='simulations (BB)')
plt.xlim(2, int(ellmax))
plt.ylim(1.e-5, 10.)
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_{\ell} \, [\mu {\rm K}^2]$', fontsize=16)
plt.grid()
plt.legend(loc='upper right', ncol=1, fontsize=9)
plt.savefig(sim_dir+sim_theory+'_Nsim'+str(Nsim)+'_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Cl_convtheory'+'_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)

# Plot results - pure
plt.clf()
plt.semilogy(ell, EEDust[0:int(ellmax)+1], 'k', lw=1.5, alpha=0.7, label='input theory (EE)')
plt.semilogy(ell, BBDust[0:int(ellmax)+1], 'k', lw=1.5, alpha=0.7, ls='--', label='input theory (BB)')
plt.semilogy(ell_binned, ClDust_binned_pure[0], 'bo', label='convolved input theory (EE)')
plt.semilogy(ell_binned*1.1, ClDust_binned_pure[3], 'mo', label='convolved input theory (BB)')
plt.errorbar(ell_binned, ClEE_pure_mean, yerr=[ClEE_pure_std,ClEE_pure_std], fmt='c', ecolor='c', elinewidth=1.5, capsize=3, capthick=1, marker='.', label='simulations (EE)')
plt.errorbar(ell_binned*1.1, ClBB_pure_mean, yerr=[ClBB_pure_std,ClBB_pure_std], fmt='g', ecolor='g', elinewidth=1.5, capsize=3, capthick=1, marker='.', label='simulations (BB)')
plt.xlim(2, int(ellmax))
plt.ylim(1.e-5, 10.)
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_{\ell} \, [\mu {\rm K}^2]$', fontsize=16)
plt.grid()
plt.legend(loc='upper right', ncol=1, fontsize=9)
plt.savefig(sim_dir+sim_theory+'_Nsim'+str(Nsim)+'_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_convtheory'+'_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
