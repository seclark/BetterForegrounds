import numpy as np
import glob, pickle
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
from subprocess import call, PIPE
from astropy.io import fits

Nside=2048
Npix=12*Nside**2

# output placement
out_root = '../data/'

# hp projected RHT angles in Equatorial coordinates
QURHT_root = "/Volumes/DataDavy/GALFA/SC_241/thetarht_maps/Planck_projected/"
Qdata = fits.getdata(QURHT_root+"Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected.fits")
Udata = fits.getdata(QURHT_root+"U_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected.fits")

# translate from IAU + B-field to Planck + dust pol
TQUmap = np.zeros((3,Npix))
TQUmap[1] = -Qdata # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT 
TQUmap[2] = Udata # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT

# make placeholder TQU map - this is NEST ordered, Equatorial coordinates
hp.fitsfunc.write_map(out_root+'/temp.fits', TQUmap, coord='C') #have to save map to use with f90 healpix utilities

# change to RING ordered
ring_indx = hp.nest2ring(Nside, np.arange(Npix))
for _tqu in range(3):
    TQUmap[_tqu, :] = TQUmap[_tqu, ring_indx]
    
print('NEST converted to RING')

# convert the TQU map to alm^TEB using anafast
call("/Users/susanclark/Healpix_3.30/bin_gfortran/anafast anafast_paramfile_S.txt", shell=True, stdout=PIPE)

# - rotate the alm^TEB from Equ to Gal coords using alteralm
call("/Users/susanclark/Healpix_3.30/bin_gfortran/alteralm alteralm_paramfile_S.txt", shell=True, stdout=PIPE)

# - convert the rotated alm^TEB back to a real-space TQU map in new coords
call("/Users/susanclark/Healpix_3.30/bin_gfortran/synfast synfast_paramfile_S.txt", shell=True, stdout=PIPE)

# - save the resulting map, so we'll have them for later use in making interpolating function
TQUmapGal = np.zeros((3,Npix))
TQUmapGal[0], TQUmapGal[1], TQUmapGal[2] = hp.fitsfunc.read_map(out_root+'/temp_Gal.fits', field=(0,1,2))
hp.fitsfunc.write_map(out_root+'/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Equ_inGal.fits', TQUmapGal, coord='G')

# remove temp files
call("ls /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)
call("rm /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)

# plot theta map
thetaGal = np.mod(0.5*np.arctan2(TQUmapGal[2], TQUmapGal[1]), np.pi)
hp.mollview(thetaGal, unit='rad', title='theta_RHT_Equ_inGal', min=0.0, max=np.pi, coord='G')
    
