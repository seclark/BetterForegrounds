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
mask_dir = '/data/jch/Planckdata/'
mask_name = 'HFI_Mask_GalPlane-apo0_2048_R2.00'
mask_field = 3 #GAL070
# apodization scale
apod_Arcmin = 60.
apod_Deg = apod_Arcmin/60.
# mask file name
mask_file = mask_dir+mask_name+FITS_end
# read in mask and apodize
# to use the pure-B formalism, need the mask to be differentiable at the boundary; the "C1" and "C2" schemes in nmt satisfy this criterion
mask = hp.read_map(mask_file, verbose=False, field=mask_field)
#print float(np.sum(mask))/float(Npix) #check fsky
apod_Type = 'C2'
#print "apodizing mask"
mask_apod = nmt.mask_apodization(mask, apod_Deg, apotype=apod_Type)
#print float(np.sum(mask_apod))/float(Npix) #check apodized fsky
#print "done apodizing"
#####

#####
# data directory and maps
data_dir = '/data/jch/Planckdata/'
data_name = 'HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full'
#data_name2 = 'HFI_SkyMap_353-psb-field-IQU_2048_R3.00_full'
#data_name = 'HFI_SkyMap_217-field-IQU_2048_R3.00_full'
data_name2 = 'HFI_SkyMap_217-field-IQU_2048_R3.00_full'
ellmax = 1001
ell = np.arange(int(ellmax)+1)
# beams, if needed
FWHM_217 = 5.02 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
FWHM_353 = 4.94 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
bl_217 = (hp.gauss_beam(fwhm=FWHM_217*(np.pi/180.0/60.0), lmax=ellmax, pol=True))[:,1] # extract polarized beam (this is EE, should be identical for BB)
bl_353 = (hp.gauss_beam(fwhm=FWHM_353*(np.pi/180.0/60.0), lmax=ellmax, pol=True))[:,1] # extract polarized beam (this is EE, should be identical for BB)
#####

#####
# define binning scheme
binwidth = 20
bins = nmt.NmtBin(N_side, nlb=binwidth, lmax=int(ellmax))
ell_binned = bins.get_effective_ells()
nbins = len(ell_binned)
#####

#####
# read in the maps and compute the mode-coupling matrix for non-pure and pure fields
Q_353, U_353 = hp.read_map(data_dir+data_name+FITS_end, field=[1, 2], verbose=True)
Q_217, U_217 = hp.read_map(data_dir+data_name2+FITS_end, field=[1, 2], verbose=True)
# non-pure
EB_npure_353 = nmt.NmtField(mask_apod, [Q_353,U_353])
EB_npure_217 = nmt.NmtField(mask_apod, [Q_217,U_217])
#print "mask and maps read in"
w_npure = nmt.NmtWorkspace()
#print "computing coupling matrix"
w_npure.compute_coupling_matrix(EB_npure_353, EB_npure_217, bins)
#print "coupling matrix done"
# pure
EB_pure_353 = nmt.NmtField(mask_apod, [Q_353,U_353], purify_e = True, purify_b = True)
EB_pure_217 = nmt.NmtField(mask_apod, [Q_217,U_217], purify_e = True, purify_b = True)
#print "mask and maps read in (pure)"
w_pure = nmt.NmtWorkspace()
#print "computing coupling matrix (pure)"
w_pure.compute_coupling_matrix(EB_pure_353, EB_pure_217, bins)
#print "coupling matrix done (pure)"
#####

#####
# get theory predictions with mode-coupling accounted for
#ClDust_binned_npure = w_npure.decouple_cell(w_npure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
#ClDust_binned_pure = w_pure.decouple_cell(w_pure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
#####

#####
# non-pure
Cl_2x2 = w_npure.decouple_cell(nmt.compute_coupled_cell(EB_npure_353,EB_npure_217)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
ClEE = Cl_2x2[0]
ClBB = Cl_2x2[3]
# pure
Cl_2x2_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_pure_353,EB_pure_217)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
ClEE_pure = Cl_2x2_pure[0]
ClBB_pure = Cl_2x2_pure[3]
#####

#####
# Plot results - non-pure
plt.clf()
plt.title('353 GHz x 217 GHz (beams not deconvolved; non-pure estimator)')
plt.semilogy(ell_binned, ClEE, 'bo', label='EE')
plt.semilogy(ell_binned*1.1, ClBB, 'mo', label='BB')
plt.xlim(2, int(ellmax))
#plt.ylim(1.e-5, 10.)
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_{\ell} \, [{\rm K}^2]$', fontsize=16)
plt.grid()
plt.legend(loc='upper right', ncol=1, fontsize=9)
plt.savefig(data_dir+'353x217_R3.00_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Cl_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
np.savetxt(data_dir+'353x217_R3.00_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Cl_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, ClEE, ClBB])))

# Plot results - pure
plt.clf()
plt.title('353 GHz x 217 GHz (beams not deconvolved; pure estimator)')
plt.semilogy(ell_binned, ClEE_pure, 'bo', label='EE')
plt.semilogy(ell_binned*1.1, ClBB_pure, 'mo', label='BB')
plt.xlim(2, int(ellmax))
#plt.ylim(1.e-5, 10.)
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_{\ell} \, [{\rm K}^2]$', fontsize=16)
plt.grid()
plt.legend(loc='upper right', ncol=1, fontsize=9)
plt.savefig(data_dir+'353x217_R3.00_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
np.savetxt(data_dir+'353x217_R3.00_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, ClEE_pure, ClBB_pure])))

