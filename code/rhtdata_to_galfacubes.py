from __future__ import division
import glob
import numpy as np
import cPickle
import time
import copy
import os.path
import cPickle as pickle
from astropy.io import fits
from astropy import wcs
from scipy import ndimage
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
import sqlite3
import healpy as hp
import itertools
import string
from reproject import reproject_interp
 
 
path_to_galfa_cubes = "/disks/jansky/a/users/goldston/DR2W_RC5/Wide/"
path_to_rht_thetaslices = "/disks/jansky/a/users/goldston/susan/Wide_maps/"

# eventually we will step through these cubes -- start w/ 1 test
galfa_cube_name = "GALFA_HI_RA+DEC_356.00+34.35_W"
galfa_cube_fn = path_to_galfa_cubes + galfa_cube_name + ".fits"

galfa_cube_data = fits.getdata(galfa_cube_fn)
galfa_cube_hdr = fits.getheader(galfa_cube_fn)

nthets = 165
rht_data_cube = np.zeros((nthets, galfa_cube_hdr['NAXIS2'], galfa_cube_hdr['NAXIS1']), np.float_)

# construct a 2D header from galfa cube to project each theta slice to
hdulist_todata = fits.open(galfa_cube_fn)
hdu_todata = hdulist_todata[0]
hdu_todata.header.remove('CRPIX3') # remove all 3rd axis keywords from fits header
hdu_todata.header.remove('CTYPE3')
hdu_todata.header.remove('CRVAL3')
hdu_todata.header.remove('CDELT3')
hdu_todata.header.remove('NAXIS3')
hdu_todata.header.remove('CROTA3')
hdu_todata.header['NAXIS'] = 2
#hdu_todata.data = galfa_cube_data[0, :, :]

new_header = fits.getheader(hdu_todata)#copy.copy(hdu_todata.header)

print(new_header)

for thet_i in xrange(nthets):
    allsky_fn = path_to_rht_thetaslices + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_"+str(thet_i)+".fits"
    allsky_thetaslice_data = fits.getdata(allsky_fn)
    allsky_thetaslice_hdr = fits.getheader(allsky_fn)
    
    print(allsky_thetaslice_data.shape)
    print(allsky_thetaslice_hdr)
    
    # Reproject each theta slice into appropriate theta bin in cube
    output, footprint = reproject_interp((allsky_thetaslice_data, allsky_thetaslice_hdr), new_header)
    print(output.shape)
    print(rht_data_cube.shape)
    rht_data_cube[thet_i, :, :] = output

new_hdr = copy.copy(galfa_cube_hdr)    
new_hdr['NAXIS3'] = nthets
new_hdr['CTYPE3'] = 'THETARHT'
new_hdr['CRVAL3'] = 0.000000
new_hdr['CRPIX3'] = 0.000000
new_hdr['CDELT3'] = np.pi/nthets

out_fn = path_to_galfa_cubes + galfa_cube_name + "_RHT.fits"
fits.writeto(out_fn, rht_data_cube, header = new_hdr)




