from __future__ import division, print_function
import numpy as np
import healpy as hp
from bayesian_machinery import *
from plot_map_outcomes import nonzero_data


# planck dbs
planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
planck_tqu_cursor = planck_tqu_db.cursor()
planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
planck_cov_cursor = planck_cov_db.cursor()

# rht data
rht_cursor, tablename = get_rht_cursor()
allids = get_all_rht_ids(rht_cursor, tablename)

# planck angles from flat-prior
p_planck_flatprior_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
psi_planck_flatprior_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
p_planck_flatprior = hp.fitsfunc.read_map(p_planck_flatprior_fn)
psi_planck_flatprior = hp.fitsfunc.read_map(psi_planck_flatprior_fn)

# angles after RHT prior is applied (legit process data)
pMLmap_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"
psiMLmap_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"
p_planckRHT_ML = hp.fitsfunc.read_map(pMLmap_fn)
psi_planckRHT_ML = hp.fitsfunc.read_map(psiMLmap_fn)

# Planck angles in original form
planck353_fn = "/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits"
planckT, planckQ, planckU = hp.fitsfunc.read_map(planck353_fn, field=(0,1,2))
#psi = np.mod(0.5*np.arctan2(planckU, -planckQ), np.pi) #Q -> -Q to change Healpix pol angle -> IAU B-field angle
psi = np.mod(0.5*np.arctan2(planckU, planckQ), np.pi) # Everything should be done in Planck, Galactic, pol angle space
planckp_orig = np.sqrt(planckQ**2 + planckU**2)/planckT

# RHT angles in *galactic* coordinates
#RHT_TQU_Gal_fn = "../data/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Equ_inGal_mask.fits"
RHT_TQU_Gal_fn = "../data/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Planck_pol_ang_Gal_mask.fits"
RHT_T_Gal, QRHT, URHT = hp.fitsfunc.read_map(RHT_TQU_Gal_fn, field=(0,1,2))
intrhtmask = np.zeros(RHT_T_Gal.shape)
intrhtmask[RHT_T_Gal > 0.5] = 1
theta_rht = np.mod(0.5*np.arctan2(URHT, QRHT), np.pi) # already in Planck, Galactic, pol angle space

# RHT angles in galactic coords, take 2
RHT_TQU_Gal_fn_2 = "/Volumes/DataDavy/Foregrounds/RHTmaps/TQU_RHT_Planck_pol_ang_SC_241_Gal.fits"
RHT_T_Gal2, QRHT2, URHT2 = hp.fitsfunc.read_map(RHT_TQU_Gal_fn_2, field=(0,1,2))
intrhtmask2 = np.zeros(RHT_T_Gal2.shape)
intrhtmask2[RHT_T_Gal2 > 0.5] = 1
theta_rht2 = np.mod(0.5*np.arctan2(URHT2, QRHT2), np.pi) # already in Planck, Galactic, pol angle space

# angles from MB-sampled normed prior
psi_prior_MB = hp.fitsfunc.read_map('../data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_deltafuncprior_False_testpsiproj_True.fits')

# small offset data
psi_prior_smalloffset = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_deltafuncprior_False_testpsiproj_True_smalloffset.fits')

# angles from pre- and post-roll thetaRHT
preroll_thetas = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/RHTmaps/preroll_thetaRHTs.fits')
postroll_thetas = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/RHTmaps/postroll_thetaRHTs.fits')

# check the above against map rotated by Colin's friend's code:
#RHT_TQU_Gal_fn_new = "/Volumes/DataDavy/Foregrounds/RHTmaps/TQU_RHT_Planck_pol_ang_SC_241_Gal.fits"
#RHT_T_Gal_new, QRHT_new, URHT_new = hp.fitsfunc.read_map(RHT_TQU_Gal_fn, field=(0,1,2))
# check passed: they are identical.

planckpsi, planckpsiplot = nonzero_data(psi, mask=intrhtmask, samesize=True)
rhtpsi, rhtpsiplot = nonzero_data(theta_rht, mask=intrhtmask, samesize=True)
planckp, planckpplot = nonzero_data(planckp_orig, mask=intrhtmask, samesize=True)
planckRHTpriorpsi, planckRHTpriorpsiplot = nonzero_data(psi_planckRHT_ML, mask=intrhtmask, samesize=True)
planckRHTpriorp, planckRHTpriorpplot = nonzero_data(p_planckRHT_ML, mask=intrhtmask, samesize=True)
psi_prior_MB, psi_prior_MBplot = nonzero_data(psi_prior_MB, mask=intrhtmask, samesize=True)
psi_prior_smalloffset, psi_prior_smalloffsetplot = nonzero_data(psi_prior_smalloffset, mask=intrhtmask, samesize=True)
preroll_thetas, preroll_thetasplot = nonzero_data(preroll_thetas, mask=intrhtmask, samesize=True)
postroll_thetas, postroll_thetasplot = nonzero_data(postroll_thetas, mask=intrhtmask, samesize=True)

# First test: are the indices the same?
planckpsi_zero = copy.copy(planckpsiplot)
planckpsi_zero[np.where(np.isnan(planckpsiplot))] = 0.0  
rawindices = np.nonzero(planckpsi_zero)

# masked data for histograms
psi_planck_flatprior_mask = copy.copy(psi_planck_flatprior)
psi_planck_flatprior_mask[np.where(np.isnan(planckpsiplot) == True)] = None
p_planck_flatprior_mask = copy.copy(p_planck_flatprior)
p_planck_flatprior_mask[np.where(np.isnan(planckpsiplot) == True)] = None

# sigpGsq data
sigpGsq = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/planck_sigpGsq_SC_241.fits")
sigpGsq_0_indx = np.where(sigpGsq == 0)
sigpGsq[sigpGsq_0_indx] = None
sigpGsq[np.where(np.isnan(planckpsiplot) == True)] = None

planckpsiplot[sigpGsq_0_indx] = None
psi_planck_flatprior_mask[sigpGsq_0_indx] = None
planckpplot[sigpGsq_0_indx] = None
p_planck_flatprior_mask[sigpGsq_0_indx] = None
planckRHTpriorpsiplot[sigpGsq_0_indx] = None
planckRHTpriorpplot[sigpGsq_0_indx] = None
psi_prior_MBplot[sigpGsq_0_indx] = None
psi_prior_smalloffsetplot[sigpGsq_0_indx] = None
preroll_thetasplot[sigpGsq_0_indx] = None
postroll_thetasplot[sigpGsq_0_indx] = None

sigpG = np.sqrt(sigpGsq)

p_over_sigma = planckpplot/sigpG
p_over_sigma_lt_1 = np.where(p_over_sigma < 1)
p_over_sigma_gt_1_lt_2 = np.where((p_over_sigma > 1) & (p_over_sigma < 2))
p_over_sigma_gt_2 = np.where(p_over_sigma > 2)

# they are not. they are similar, but not the same.
Nside=2048
Npix=12*Nside**2
indices_allids = np.zeros(Npix, np.int_)
indices_planck = np.zeros(Npix, np.int_)

allids_ring = hp.pixelfunc.nest2ring(Nside, allids)
indices_allids_ring = np.zeros(Npix, np.int_)
indices_allids_ring[allids_ring] = 1

indices_allids[allids] = 1
indices_planck[rawindices] = 1

# They're off by having some pixels lit in one that aren't in the other. Sparse differences - not urgent.
#In [97]: np.nansum(indices_allids_ring - indices_planck)
#Out[97]: 53735

# However, note that allids are in *nest* while indices_planck are in *ring* until changed.

# The challenge now is to pick a given pixel, grab its Planck likelihood, RHT prior, and posterior, and
# see whether these correspond to the expected Planck and RHT 'raw' values.

# grab some id. these are in nested order
#id = allids[1000][0]
#id = 3662707
#id_ring = hp.pixelfunc.nest2ring(Nside, id)

# this is an id that only one pixel in GALFA-HI contributed to in the interpolated hp map:
id_ring = 18691216#22123685
id = hp.pixelfunc.ring2nest(Nside, id_ring)

pp = Posterior(id, rht_cursor=rht_cursor, adaptivep0=False)
pp_adaptive = Posterior(id, rht_cursor=rht_cursor, adaptivep0=True)
axs = plot_all_bayesian_components_from_posterior(pp, returnax=True)

# test whether projected theta_rht is correct theta_rht
rht_data = pp.prior_obj.rht_data
sample_psi0 = pp.prior_obj.sample_psi0

QRHT_from_proj_bins = np.sum(np.cos(2*sample_psi0)*rht_data)
URHT_from_proj_bins = np.sum(np.sin(2*sample_psi0)*rht_data)
theta_rht_from_proj_bins = np.mod(0.5*np.arctan2(URHT_from_proj_bins, QRHT_from_proj_bins), np.pi)

posterior_is_prior = copy.copy(pp)
posterior_is_prior.normed_posterior = pp.normed_prior
MB_p_prior, MB_psi_prior = mean_bayesian_posterior(posterior_is_prior)

MB_p_posterior, MB_psi_posterior = mean_bayesian_posterior(pp)
MB_p_posterior_adaptive, MB_psi_posterior_adaptive = mean_bayesian_posterior(pp_adaptive)

planckposterior = PlanckPosterior(id, planck_tqu_cursor, planck_cov_cursor, pp.sample_p0, pp.sample_psi0, adaptivep0=False)
planckposterior_adaptive = PlanckPosterior(id, planck_tqu_cursor, planck_cov_cursor, pp.sample_p0, pp.sample_psi0, adaptivep0=True)

MB_p_planckonly, MB_psi_planckonly = mean_bayesian_posterior(planckposterior)
MB_p_planckonly_adaptive, MB_psi_planckonly_adaptive = mean_bayesian_posterior(planckposterior_adaptive)

print("expected planck psi = {}".format(planckpsiplot[id_ring]))
print("naive planck psi = {}".format(pp.naive_psi))
print("planck psi with flat prior = {}".format(MB_psi_planckonly))
print("planck psi with flat prior, adaptive p0 = {}".format(MB_psi_planckonly_adaptive))
print("psi with RHT prior = {}".format(MB_psi_posterior))
print("psi with RHT prior, adaptive p0 = {}".format(MB_psi_posterior_adaptive))
print("expected RHT psi = {}".format(theta_rht[id_ring]))
print("RHT-prior-only psi = {}".format(MB_psi_prior))

print("expected planck p = {}".format(planckpplot[id_ring]))
print("naive planck p = {}".format(pp.pmeas))
print("planck p with flat prior = {}".format(MB_p_planckonly))
print("planck p with flat prior, adaptive p0 = {}".format(MB_p_planckonly_adaptive))
print("p with RHT prior = {}".format(MB_p_posterior))
print("p with RHT prior, adaptive p0 = {}".format(MB_p_posterior_adaptive))


axs = plot_all_bayesian_components_from_posterior(posterior_is_prior, returnax=True)
axs[2].plot(MB_p_prior, MB_psi_prior, 'x', color='teal')
axs[2].plot(MB_p_prior, theta_rht[id_ring], 'x', color='yellow')

#MCMC_posterior(id, rht_cursor = rht_cursor, local=True)

# histogram of p
print(len(planckpplot[~np.isnan(planckpplot)]), len(p_planck_flatprior_mask[~np.isnan(p_planck_flatprior_mask)]))
plt.figure()
plt.hist(p_planck_flatprior_mask[~np.isnan(p_planck_flatprior_mask)], bins=100, histtype='step', range=(-0.5,2.0))
plt.hist(planckpplot[~np.isnan(planckpplot)], bins=100, histtype='step', range=(-0.5,2.0))
plt.legend(['flatprior', 'naive'])
plt.title('p planck hist')

plt.figure()
plt.hist(psi_planck_flatprior_mask[~np.isnan(psi_planck_flatprior_mask)], bins=100, histtype='step')
plt.hist(planckpsiplot[~np.isnan(planckpsiplot)], bins=100, histtype='step')
plt.hist(planckRHTpriorpsiplot[~np.isnan(planckRHTpriorpsiplot)], bins=100, histtype='step')
plt.legend(['flatprior', 'naive', 'RHT prior'])
plt.title('psi planck hist')

# compare psi from prior only to projected RHT psi (galactic)
plt.figure()
plt.hist(rhtpsiplot[~np.isnan(rhtpsiplot)], bins=100, histtype='step')
plt.hist(psi_prior_MBplot[~np.isnan(psi_prior_MBplot)], bins=100, histtype='step')
plt.hist(psi_prior_smalloffsetplot[~np.isnan(psi_prior_smalloffsetplot)], bins=100, histtype='step')
plt.hist(preroll_thetasplot[~np.isnan(preroll_thetasplot)], bins=100, histtype='step')
#plt.hist(postroll_thetasplot[~np.isnan(postroll_thetasplot)], bins=100, histtype='step')
plt.legend([r'rotated from $\hat{\theta}_{RHT}$', 'prior MB large Z', 'prior MB small Z', r'$\hat{\psi}_{RHT}$'])

# histogram of sigpGsq
plt.figure()
plt.hist(sigpGsq[~np.isnan(sigpGsq)], bins=100, histtype='step', range=(0, 1))

plt.hist(planckpplot[~np.isnan(planckpplot)]/sigpG[~np.isnan(sigpG)], bins=100, histtype='step')#, range=(0, 1))

fig = plt.figure()
hp.mollview(p_planck_flatprior_mask, sub=(222), title='p, flat prior')
hp.mollview(planckpplot, sub=(221), title='p, naive')
hp.mollview(psi_planck_flatprior_mask, sub=(224), title=r'$\psi$, flat prior')
hp.mollview(planckpsiplot, sub=(223), title=r'$\psi$, naive')

fig = plt.figure()
ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)
histrange = (-0.1,2.0)
ax1.hist(planckpplot[p_over_sigma_lt_1], bins=100, histtype='step', range=histrange)
ax1.hist(p_planck_flatprior_mask[p_over_sigma_lt_1], bins=100, histtype='step', range=histrange)
ax2.hist(planckpplot[p_over_sigma_gt_1_lt_2], bins=100, histtype='step', range=histrange)
ax2.hist(p_planck_flatprior_mask[p_over_sigma_gt_1_lt_2], bins=100, histtype='step', range=histrange)
ax3.hist(planckpplot[p_over_sigma_gt_2], bins=100, histtype='step', range=histrange)
ax3.hist(p_planck_flatprior_mask[p_over_sigma_gt_2], bins=100, histtype='step', range=histrange)
ax3.legend(['naive', 'flat prior'])
ax1.title(r'$\sigma_{pG} < 1$')
ax2.title(r'$1 < \sigma_{pG} < 2$')
ax3.title(r'$\sigma_{pG} > 2$')




