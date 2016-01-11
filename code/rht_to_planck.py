from __future__ import division, print_function
import numpy as np
import healpy as hp
import math
from astropy.io import fits

def get_thets(wlen, save = False):
    """
    Theta bins for a given rolling window length.
    Formula in Clark+ 2014
    """
    ntheta = math.ceil((np.pi*np.sqrt(2)*((wlen-1)/2.0)))
    print('ntheta is ', ntheta)
    dtheta = np.pi/ntheta
    
    # Thetas for binning (edges of bins)   
    thetbins = dtheta*np.arange(0, ntheta+2)
    thetbins = thetbins - dtheta/2
    
    # Thetas for plotting (centers of bins)
    thets = np.arange(0, np.pi, dtheta)
    
    if save == True:
        np.save('/Volumes/DataDavy/GALFA/original_fibers_project/thets_w'+str(wlen)+'.npy', thets)
    
    return thets
    
# Our analysis is run with wlen = 75
wlen = 75
thets = get_thets(wlen)

# full-galfa-sky file
fgs_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/GALFA_HI_W_S1019_1023.fits"
fgs_hdr = fits.get_header(fgs_fn)

# Fill array with each individual theta
blank = np.zeros((fgs_hdr["NAXIS1"], fgs_hdr["NAXIS2"], len(thets), np.float_)
for i in xrange(len(thets)):
    blank[:, :, i] = thets[i]

