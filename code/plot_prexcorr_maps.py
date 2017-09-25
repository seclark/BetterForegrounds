import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

maproot = "/Volumes/DataDavy/Foregrounds/BayesianMaps/"


p_naive=hp.fitsfunc.read_map(maproot+"pMB_SC_241_353GHz_naive_abspsidx.fits")
psi_naivep=hp.fitsfunc.read_map(maproot+"psiMB_SC_241_353GHz_naive_abspsidx.fits")

p_thetrht1=hp.fitsfunc.read_map(maproot+"pMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits")
psi_thetrht1=hp.fitsfunc.read_map(maproot+"psiMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits")

p_bamp0=hp.fitsfunc.read_map(maproot+"pMB_DR2_SC_241_-10_10_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits")
psi_bamp0=hp.fitsfunc.read_map(maproot+"psiMB_DR2_SC_241_-10_10_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits")

p_bamp1=hp.fitsfunc.read_map(maproot+"pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_1.fits")
psi_bamp1=hp.fitsfunc.read_map(maproot+"psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_1.fits")

fig = plt.figure()

plotmaps = [p_naive, p_thetrht1, p_bamp0, p_bamp1]
plotmaps = [psi_naivep, psi_thetrht1, psi_bamp0, psi_bamp1]
plottitles = ["p naive", "p thetrht1", "p bamp0", "p bamp1"]
plottitles = [r"$\psi$ naive", r"$\psi$ thetrht1", r"$\psi$ bamp0", r"$\psi$ bamp1"]
axs = [ax1, ax2, ax3, ax4]
subnums = [221, 222, 223, 224]

for i, (plotmap, ax, subnum, plottitle) in enumerate(zip(plotmaps, axs, subnums, plottitles)):
    hp.mollview(plotmap, sub=subnum, title=plottitle)
    
T353map, Q353map, U353map=hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits", field=(0,1,2))
t, q_bamp0, u_bamp0 = hp.fitsfunc.read_map(maproot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_"+"bamp0"+".fits", field=(0,1,2))
t, q_bamp0p01, u_bamp0p01 = hp.fitsfunc.read_map(maproot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_"+"bamp1E-2"+".fits", field=(0,1,2))
t, q_bamp1, u_bamp1 = hp.fitsfunc.read_map(maproot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_"+"baseamp1"+".fits", field=(0,1,2))
t, q_thetrht1, u_thetrht1 = hp.fitsfunc.read_map(maproot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_"+"thetrht1"+".fits", field=(0,1,2))
t, q_rht1mmax, u_rht1mmax = hp.fitsfunc.read_map(maproot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_"+"rht1mmax"+".fits", field=(0,1,2))

weightsA = hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_zerod.fits")

plotmaps=[Q353map, q_bamp1, q_bamp0p01, q_bamp0]
plotmaps=[U353map, u_bamp1, u_bamp0p01, u_bamp0]
plottitles=["353 raw", "Z=1", "Z=1E-2", "Z=0"]
subnums = [221, 222, 223, 224]

fig = plt.figure()
for i, (plotmap, subnum, plottitle) in enumerate(zip(plotmaps, subnums, plottitles)):
    plotmap[weightsA < 0.5] = None
    hp.cartview(plotmap, latra=(5, 90), lonra=(-30, 110), sub=subnum, title=plottitle)
   # hp.mollview(plotmap, sub=subnum, title=plottitle)
    
    
RHT_TQU_Gal_fn = "../data/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Gal.fits"
RHT_T_Gal, QRHT, URHT = hp.fitsfunc.read_map(RHT_TQU_Gal_fn, field=(0,1,2))

psi_planck_flatprior_fn = maproot+"/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
psi_planck_flatprior = hp.fitsfunc.read_map(psi_planck_flatprior_fn)
psi_planck_flatprior_mask = copy.copy(psi_planck_flatprior)
psi_planck_flatprior_mask[weightsA < 0.5] = None

plotmaps=[thetarhtgal_mask, thet353_mask, thet1mmax]
subnums = [131, 132, 133]
plottitles=[r'$\psi_{RHT}$', r'$\psi_{353}$', r'1-R($\psi$) prior']
fig = plt.figure()
for i, (plotmap, subnum, plottitle) in enumerate(zip(plotmaps, subnums, plottitles)):
    plotmap[weightsA < 0.5] = None
    hp.cartview(plotmap, latra=(5, 90), lonra=(-30, 110), sub=subnum, title=plottitle)

plotmaps=[thetarhtgal_mask, thet353_mask, psi_planck_flatprior_mask, thet1mmax]
fig = plt.figure()
for plotmap in plotmaps:
    out = plt.hist(plotmap[~np.isnan(plotmap)], bins=100, histtype='step')
plt.legend([r'$\psi_{RHT}$', r'$\psi_{353}$', r'$\psi^{MB}_{353} flat prior$', r'1-R($\psi$) prior'])
