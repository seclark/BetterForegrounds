import numpy as np
import glob, pickle
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
from subprocess import call, PIPE
from astropy.io import fits

import rht_to_planck
import rotate_map_alm

Nside=2048
Npix=12*Nside**2

# output placement
out_root = '../data/'

# hp projected RHT angles in Equatorial coordinates
QURHT_root = "/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/Planck_projected/"
Qdata_fn = QURHT_root+"Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected.fits"
Udata_fn = QURHT_root+"U_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected.fits"

projectGal = True
# these are unprojected maps that we can project to Galactic coordinates
if projectGal:
    #QURHT_root = "/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/"
    #Qdata_fn = QURHT_root+"Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.fits"
    #Udata_fn = QURHT_root+"U_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr.fits"
    QURHT_root ="/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/QUmaps/"
    Qdata_fn = QURHT_root+"QRHT_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70.fits"
    Udata_fn = QURHT_root+"URHT_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70.fits"

    Qdata = fits.getdata(Qdata_fn)
    Udata = fits.getdata(Udata_fn)

    # get header and make truly 2-dimensional
    hdulist = fits.open(Qdata_fn)
    hdu = hdulist[0]
    #hdu.header.remove('CRPIX3') # remove all 3rd axis keywords from fits header
    #hdu.header.remove('CTYPE3')
    #hdu.header.remove('CRVAL3')
    #hdu.header.remove('CDELT3')
    #hdu.header.remove('CROTA3')

    # transform to Galactic and translate from IAU + B-field to Planck + dust pol
    # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT 
    Qdata_Equ, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(Qdata, hdu.header, local=True, Equ=True) # no conversion
    Udata_Equ, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(Udata, hdu.header, local=True, Equ=True) 
else: 
    Qdata_Equ = fits.getdata(Qdata_fn)
    Udata_Equ = fits.getdata(Udata_fn)

# TQU map of Equatorial coordinate data
TQUmap = np.zeros((3,Npix))
TQUmap[0][np.where(Qdata_Equ != -999)] = 1
TQUmap[1] = -Qdata_Equ  # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT
TQUmap[2] = Udata_Equ 

# mask instead of -999
TQUmap[1, np.where(Qdata_Equ == -999)] = None
TQUmap[2, np.where(Udata_Equ == -999)] = None



# change to RING ordered
for _tqu in range(3):
    TQUmap[_tqu, :] = hp.reorder(TQUmap[_tqu, :], n2r=True)
    
print('NEST converted to RING')

#testmap = rotate_map_alm.rotate_map(TQUmap, 2000.0,2000.0,'C','G',Nside)
#hp.fitsfunc.write_map(out_root+'/testmap.fits', testmap, coord='G') #save to compare

    
# make placeholder TQU map - this is RING ordered, Equatorial angle, Galactic coordinates
#hp.fitsfunc.write_map('/Volumes/DataDavy/Foregrounds/RHTmaps/TQU_RHT_Planck_pol_ang_SC_241_Equ.fits', TQUmap, coord='C')
#hp.fitsfunc.write_map('/Volumes/DataDavy/Foregrounds/RHTmaps/TQU_RHT_Planck_pol_ang_SC_241_Equ_thets1.fits', TQUmap, coord='C')
hp.fitsfunc.write_map('/Volumes/DataDavy/Foregrounds/RHTmaps/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Equ.fits', TQUmap, coord='C')
hp.fitsfunc.write_map(out_root+'/temp.fits', TQUmap, coord='C') #have to save map to use with f90 healpix utilities


# convert the TQU map to alm^TEB using anafast
call("/Users/susanclark/Healpix_3.30/bin_gfortran/anafast anafast_paramfile_S.txt", shell=True, stdout=PIPE)

# - rotate the alm^TEB from Equ to Gal coords using alteralm
call("/Users/susanclark/Healpix_3.30/bin_gfortran/alteralm alteralm_paramfile_S.txt", shell=True, stdout=PIPE)

# - convert the rotated alm^TEB back to a real-space TQU map in new coords
call("/Users/susanclark/Healpix_3.30/bin_gfortran/synfast synfast_paramfile_S.txt", shell=True, stdout=PIPE)

# - save the resulting map, so we'll have them for later use in making interpolating function
TQUmapGal = np.zeros((3,Npix))
TQUmapGal[0], TQUmapGal[1], TQUmapGal[2] = hp.fitsfunc.read_map(out_root+'/temp_Gal.fits', field=(0,1,2))
#hp.fitsfunc.write_map(out_root+'/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Planck_pol_ang_Gal_mask.fits', TQUmapGal, coord='G')
#hp.fitsfunc.write_map(out_root+'/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Planck_pol_ang_Gal_mask_thets1.fits', TQUmapGal, coord='G')
hp.fitsfunc.write_map(out_root+'/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Gal.fits', TQUmapGal, coord='G')
# remove temp files
call("ls /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)
call("rm /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)

# plot theta map
#thetaGal = np.mod(0.5*np.arctan2(TQUmapGal[2], TQUmapGal[1]), np.pi)
#hp.mollview(thetaGal, unit='rad', title='theta_RHT_Equ_inGal', min=0.0, max=np.pi, coord='G')
    
