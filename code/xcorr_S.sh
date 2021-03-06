#! /bin/bash
# for polspice
#HEALPIXDATA=/Users/susanclark/Healpix_3.30/data
#HEALPIX=/Users/susanclark/Healpix_3.30
#SPICE=/Users/susanclark/PolSpice_v03-00-03/src/spice
HEALPIXDATA=/home/seclark/Healpix_3.31/data
HEALPIX=/home/seclark/Healpix_3.31
SPICE=/home/seclark/PolSpice_v03-04-01/src/spice

#testname=test1
#testname=AvM
#testname=SC-4_3_g3
#testname=p353s
#testname=p353s2
#testname=adaptivep0
#testname=padaptp0
#testname=padaptp02
#testname=ptol0
#testname=rhttol0
#testname=ptol13
#testname=rhttol13
#testname=pMAP
#testname=rhtMAP
#testname=mynewMB
#testname=pnewMB
#testname=fixpsihack
#testname=revrht
#testname=newplanck
#testname=baseamp1
#testname=bamp1E-2
#testname=pnaive
#testname=bamp0
#testname=smoothb0
#testname=b0-10_10
#testname=thetrht1
#testname=thetsig30
#testname=r1mmaxpat
#testname=rhtmedvar
#testname=rhtmaxvar
#testname=max10ad
#testname=med10ad
#testname=avmfixw
#testname=b0MAP
testname=weight0

# map locations
#T353map=/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits #T is field 0, Q is field 1, U is field 2
T353map=/Users/susanclark/Dropbox/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits #T is field 0, Q is field 1, U is field 2
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_test0.fits # original tests with R(theta)
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_test0.fits
#pMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/pMB_SC_241_thetaRHT_test0_RING.fits
#psiMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/psiMB_SC_241_thetaRHT_test0_RING.fits
#pMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/pMB_SC_241_thetaRHT_AvM.fits
#psiMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/psiMB_SC_241_thetaRHT_AvM.fits
#pMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/pMB_DR2_SC_241_-4_3_smoothprior_True.fits
#psiMLmap=/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/psiMB_DR2_SC_241_-4_3_smoothprior_True.fits
#pMLmap=/Volumes/DataDavy/Planck/pMB_DR2_SC_241_353GHz.fits
#psiMLmap=/Volumes/DataDavy/Planck/psiMB_DR2_SC_241_353GHz.fits
#pMLmap=/Volumes/DataDavy/Planck/pMB_DR2_SC_241_353GHz_take2.fits
#psiMLmap=/Volumes/DataDavy/Planck/psiMB_DR2_SC_241_353GHz_take2.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_tol1E-10.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_tol1E-10.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_tol_0.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_tol_0.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_tol_0.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_tol_0.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_tol_0.001.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_tol_0.001.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_tol_0.001.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_tol_0.001.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_MAP_DR2_SC_241_353GHz_adaptivep0_True.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_MAP_DR2_SC_241_353GHz_adaptivep0_True.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_MAP_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_MAP_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True.fits
# my new method
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits
#pMLmap=/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits
#psiMLmap=/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/RHTmaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_2.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/RHTmaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_2.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/RHTmaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_fixedpsi0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/RHTmaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_fixedpsi0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_fixedpsi0_reverseRHT.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_fixedpsi0_reverseRHT.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_1.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_1.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.01.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.01.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_SC_241_353GHz_naive_abspsidx.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_SC_241_353GHz_naive_abspsidx.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_-10_10_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_-10_10_smoothprior_True_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_True_sig_30_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_True_sig_30_adaptivep0_False_deltafuncprior_False_baseprioramp_0.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_variable.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_variable.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_median_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_median_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_max_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-4_3_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_max_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_max_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_False_deltafuncprior_False_baseprioramp_max_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_max_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_max_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_True_adaptivep0_True_deltafuncprior_False_baseprioramp_max_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_True_adaptivep0_True_deltafuncprior_False_baseprioramp_max_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_median_var.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_RHTPrior_-10_10_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_median_var.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_True_sig_30_adaptivep0_False_fixwidth_True.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_prior_ThetaRHT_-10_10_smoothprior_True_sig_30_adaptivep0_False_fixwidth_True.fits
#pMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_MAP_DR2_SC_241_-10_10_smoothprior_False_adaptivep0_True_baseprioramp_0.fits
#psiMLmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_MAP_DR2_SC_241_-10_10_smoothprior_False_adaptivep0_True_baseprioramp_0.fits
pMLmap=/Users/susanclark/Projects/BetterForegrounds/data/pMB_DR2_SC_241_prior_RHTPrior_weighted_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_0.fits
psiMLmap=/Users/susanclark/Projects/BetterForegrounds/data/psiMB_DR2_SC_241_prior_RHTPrior_weighted_smoothprior_False_adaptivep0_True_deltafuncprior_False_baseprioramp_0.fits


#outputmap=/Users/susanclark/BetterForegrounds/data/HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_${testname}.fits
#outputmap=/Volumes/DataDavy/Foregrounds/BayesianMaps/HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_planckpatch_${testname}.fits
#TQU217maps=/Volumes/DataDavy/Planck/HFI_SkyMap_217_2048_R2.02_full_RING.fits
outputmap=/Users/susanclark/Dropbox/Foregrounds/BayesianMaps/HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_planckpatch_${testname}.fits
TQU217maps=/Users/susanclark/Dropbox/Planck/HFI_SkyMap_217_2048_R2.02_full_RING.fits

#TQU143maps=/Volumes/DataDavy/Planck/HFI_SkyMap_143_2048_R2.02_full.fits

# construct the ML template map
python TQU_setup.py $T353map $pMLmap $psiMLmap $outputmap
echo "template map constructed"

# cross-correlate with a different polarized dust tracer (e.g., 217 GHz or a different split of the 353 GHz data)
# polspice parameters -- calibrated for the mask used in CHPP15, need to be re-calibrated if mask changes
# also note use of -symmetric_cl YES in the code below (important if one wants TE, TB, and EB results)
APODSIGMA=7.65
APODTYPE=0
THETAMAX=14.0
NLMAX=1000 #maximum multipole

# input beams/weights (masks)
mapA=$outputmap
#mapA=$T353map
mapB=$TQU217maps
beamA=4.94 #Gaussian FWHM in arcmin (353 GHz)
beamB=5.02 #Gaussian FWHM in arcmin (217 GHz)
weightsA=/Users/susanclark/Dropbox/Foregrounds/RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_zerod.fits
weightsB=/Users/susanclark/Dropbox/Foregrounds/RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_zerod.fits
# output power spectra files
CLAB=cl_353full_pMB_psiMB_${testname}_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
CLAA=cl_353full_pMB_psiMB_${testname}_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
CLBB=cl_217full_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
# files for output from polspice
ABOUT=spice_output_353full_pMB_psiMB_${testname}_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt
AAOUT=spice_output_353full_pMB_psiMB_${testname}_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt
BBOUT=spice_output_217full_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt

# cross-correlation
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -beam2 $beamB -clfile $CLAB -decouple YES -mapfile $mapA -mapfile2 $mapB -weightfile $weightsA -weightfile2 $weightsB -nlmax $NLMAX -verbosity 2 -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $ABOUT
# auto-correlations
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -clfile $CLAA -decouple YES -mapfile $mapA -weightfile $weightsA -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $AAOUT
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamB -clfile $CLBB -decouple YES -mapfile $mapB -weightfile $weightsB -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $BBOUT

# bin and plot the power spectra
fskyfile=fsky_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_Equ_apodFWHM15arcmin.txt
finalfile=cl_353full_pMB_psiMB_${testname}_217full_pspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_gpixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned #.txt and .png are appended to this
#finalfile=cl_353full_pMB_psiMB_${testname}_217full_pspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_gpixcorr_UDOWN_bgt30cut_starmask_HFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned_ppatch_rht0.25 #.txt and .png are appended to this
python xcorr_bin_polspice.py $CLAB $CLAA $CLBB $fskyfile $finalfile
