#! /bin/bash
# for polspice
HEALPIXDATA=/u/jch/Healpix_2.20a/data
SPICE=/u/jch/Dropbox/Lensing_Project/projcorr_code/redshiftshellconv/sumallshells/getalm/PolSpice_v02-08-01/src/spice

# map locations
T353map=/scr/depot1/jch/Planckdata/HFI_SkyMap_353_2048_R2.02_full_RING.fits #T is field 0, Q is field 1, U is field 2
pMLmap=/scr/depot1/jch/RHT_QU/maps2016/pMB_test0.fits
psiMLmap=/scr/depot1/jch/RHT_QU/maps2016/psiMB_test0.fits
outputmap=/scr/depot1/jch/RHT_QU/maps2016/HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_test0.fits
TQU217maps=/scr/depot1/jch/Planckdata/HFI_SkyMap_217_2048_R2.02_full_RING.fits

# construct the ML template map
python TQU_setup.py $T353map $pMLmap $psiMLmap $outputmap

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
weightsA=/scr/depot1/jch/RHT_QU/RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_zerod.fits
weightsB=/scr/depot1/jch/RHT_QU/RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_zerod.fits
# output power spectra files
CLAB=cl_353full_pMB_psiMB_test0_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
CLAA=cl_353full_pMB_psiMB_test0_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
CLBB=cl_217full_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.dat
# files for output from polspice
ABOUT=spice_output_353full_pMB_psiMB_test0_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt
AAOUT=spice_output_353full_pMB_psiMB_test0_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt
BBOUT=spice_output_217full_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt

# cross-correlation
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -beam2 $beamB -clfile $CLAB -decouple YES -mapfile $mapA -mapfile2 $mapB -weightfile $weightsA -weightfile2 $weightsB -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $ABOUT
# auto-correlations
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -clfile $CLAA -decouple YES -mapfile $mapA -weightfile $weightsA -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $AAOUT
$SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamB -clfile $CLBB -decouple YES -mapfile $mapB -weightfile $weightsB -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -thetamax $THETAMAX > $BBOUT

# bin and plot the power spectra
fskyfile=fsky_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_Equ_apodFWHM15arcmin.txt
finalfile=cl_353full_pMB_psiMB_test0_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned #.txt and .png are appended to this
python xcorr_bin_polspice.py $CLAB $CLAA $CLBB $fskyfile $finalfile
