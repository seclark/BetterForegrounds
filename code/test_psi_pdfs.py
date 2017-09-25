from __future__ import division, print_function
import numpy as np
import healpy as hp
from bayesian_machinery import *
from plot_map_outcomes import nonzero_data
from astropy.io import fits

import rht_to_planck


hpnpix = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/coords/data_count_hp_projection_numpix_2.fits')
sigpGsq = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/planck_sigpGsq_SC_241.fits")
sigpGsq_0_indx = np.where(sigpGsq == 0)
sigpGsq[sigpGsq_0_indx] = None

# read in psi_hat data
#psirht = hp.fitsfunc.read_map('../data/postroll_thetaRHTs_newpsi0.fits')
psirht = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/RHTmaps/vel_-4_3_preroll_thetaRHTs_newpsi0.fits')
#psirht = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/RHTmaps/vel_-10_10_postroll_thetaRHTs_newpsi0.fits')

# read in rotated theta_hat data
#RHT_TQU_Gal_fn = "../data/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Planck_pol_ang_Gal_mask.fits"
RHT_TQU_Gal_fn = "../data/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Gal.fits"
RHT_T_Gal, QRHT, URHT = hp.fitsfunc.read_map(RHT_TQU_Gal_fn, field=(0,1,2))
intrhtmask = np.zeros(RHT_T_Gal.shape)
intrhtmask[RHT_T_Gal > 0.5] = 1
theta_rht = np.mod(0.5*np.arctan2(URHT, QRHT), np.pi) # already in Planck, Galactic, pol angle space

# original planck data
planck353_fn = "/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits"
planckT, planckQ, planckU = hp.fitsfunc.read_map(planck353_fn, field=(0,1,2))
psi = np.mod(0.5*np.arctan2(planckU, planckQ), np.pi) # Everything should be done in Planck, Galactic, pol angle space
planckpsi, planckpsiplot = nonzero_data(psi, mask=intrhtmask, samesize=True)
planck_sc241 = copy.copy(planckpsiplot)

# apply all the same masks
psirht, psirhtplot = nonzero_data(psirht, mask=intrhtmask, samesize=True)
thetarht_rot, thetarht_rotplot = nonzero_data(theta_rht, mask=intrhtmask, samesize=True)

psirhtplot[sigpGsq_0_indx] = None
thetarht_rotplot[sigpGsq_0_indx] = None
planck_sc241[sigpGsq_0_indx] = None

onepix = np.where(hpnpix == 1)
twopix = np.where(hpnpix == 2)
threepix = np.where(hpnpix == 3)
fourpix = np.where(hpnpix == 4)
fivepix = np.where(hpnpix == 5)
sixpix = np.where(hpnpix == 6)

fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)

nbins = 165
histtype = 'step'

allax = [ax1, ax2, ax3, ax4, ax5, ax6]
allnpix = [onepix, twopix, threepix, fourpix, fivepix, sixpix]

for i_, (ax, npix) in enumerate(zip(allax, allnpix)):
    psiplotme = psirhtplot[npix]
    thetarhtplotme = thetarht_rotplot[npix]
    origplanckplotme = planck_sc241[npix]
    ax.hist(psiplotme[~np.isnan(psiplotme)], bins=nbins, histtype=histtype)
    ax.hist(thetarhtplotme[~np.isnan(thetarhtplotme)], bins=nbins, histtype=histtype)
    ax.hist(origplanckplotme[~np.isnan(origplanckplotme)], bins=nbins, histtype=histtype)
    
    #angresid = polarization_tools.angle_residual(psiplotme[~np.isnan(psiplotme)], thetarhtplotme[~np.isnan(thetarhtplotme)], degrees=False)
    #ax.hist(angresid, bins=nbins, histtype=histtype)
    ax.set_title(i_ + 1)
    
#plt.suptitle(r'$\hat{\psi}$ - $\hat{\theta}_{rot}$ per maximum possible contributing pixels')

ax3.legend([r'$\hat{\psi}$', r'$\hat{\theta}_{rot}$'])
plt.suptitle('Angle histogram per maximum possible contributing pixels', y=0.999)
