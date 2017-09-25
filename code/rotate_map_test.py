import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from rotate_map_alm import *
from astropy.io import fits

Nside=2048
Npix=12*Nside**2

# Planck 353 GHz TQU
TQUmap = np.zeros((3,Npix))
#TQUmap = hp.read_map('/scr/depot1/jch/Planckdata/HFI_SkyMap_353_2048_R2.02_full_RING.fits', field=(0,1,2))
TQUfn = '/disks/jansky/a/users/goldston/susan/Planck/SOSDPol_and_HI/353GHz_IQU_2048_dipole_model_subtracted.fits'
TQUmap = hp.read_map(TQUfn, field=(0,1,2))
TQUhdr = fits.getheader(TQUfn)

# check images
for i in xrange(3):
    plt.clf()
    hp.mollview(TQUmap[i], coord='G')
    plt.savefig('rotate_map_test_Gal_'+str(i)+'.png')

# rotate
TQUmap_Equ = np.zeros((3,Npix))
TQUmap_Equ = rotate_map(TQUmap,2000.0,2000.0,'G','C',Nside)
# check images
for i in xrange(3):
    plt.clf()
    hp.mollview(TQUmap_Equ[i], coord='C')
    plt.savefig('rotate_map_test_Equ_'+str(i)+'.png')

#hp.write_map('/disks/jansky/a/users/goldston/susan/Planck/SOSDPol_and_HI/353GHz_IQU_2048_dipole_model_subtracted_Equ.fits', TQUmap_Equ)
fits.writeto('/disks/jansky/a/users/goldston/susan/Planck/SOSDPol_and_HI/353GHz_IQU_2048_dipole_model_subtracted_Equ.fits', TQUmap_Equ, TQUhdr)