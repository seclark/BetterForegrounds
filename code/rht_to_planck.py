from __future__ import division, print_function
import numpy as np
import healpy as hp
import math
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import copy

# RHT helper code
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

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

def interpolate_thetas(vstart = 1019, vstop = 1023, wlen = 75):    
    # Our analysis is run with wlen = 75
    thets = get_thets(wlen)

    # full-galfa-sky file
    root_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    fgs_fn = root_fn + "GALFA_HI_W_S"+str(vstart)+"_"+str(vstop)+".fits"
    fgs_hdr = fits.getheader(fgs_fn)

    # Planck file
    Pfile = root_fn + "HFI_SkyMap_353_2048_R2.02_full.fits"
    #"HFI_SkyMap_353_2048_R2.00_full_Equ.fits"

    # Fill array with each individual theta
    #all_bins = np.zeros((fgs_hdr["NAXIS1"], fgs_hdr["NAXIS2"], len(thets)), np.float_)
    #for i in xrange(len(thets)):
    channel_data = np.zeros((fgs_hdr["NAXIS1"], fgs_hdr["NAXIS2"]), np.float_)
    channel_data[:, :] = thets[0]

    # Planck data
    hdulist = fits.open(Pfile)
    tbdata = hdulist[1].data
    hpq = tbdata.field("Q_STOKES").flatten()
    
    gwcs = wcs.WCS(fgs_fn)
    xax = np.linspace(1, fgs_hdr["NAXIS1"], fgs_hdr["NAXIS1"]).reshape(fgs_hdr["NAXIS1"], 1)
    yax = np.linspace(1, fgs_hdr["NAXIS2"], fgs_hdr["NAXIS2"]).reshape(1, fgs_hdr["NAXIS2"])
    test = gwcs.all_pix2world(xax, yax, 1)
    RA = test[0]
    Dec = test[1]
    c = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, frame="icrs")

    cg = c.galactic

    hppos = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(50331648),  np.pi/2-np.asarray(cg.b.rad), np.asarray(cg.l.rad), nest=True)
    
    # All and final positions
    flat_hppos = hppos.flatten()
    final_data = np.zeros(hpq.size).flatten() - 999

    # Q data to place
    channel_data = ((channel_data).T)[:, :].flatten() # this should be upside down? Yes it is! So this is what we want.

    # zip position information and Qdata. 
    alldata = zip(flat_hppos, channel_data)

    # Append duplicate key entries rather than overwriting them
    grouped_data = {}
    for k, v in alldata:
        grouped_data.setdefault(k, []).append(v)

    # Average nonzero RHT data. Do not count NaN values in histogram
    for z in grouped_data.keys():
        final_data[z] = np.nansum(grouped_data[z])/np.count_nonzero(~np.isnan(grouped_data[z]))

    final_data[np.isnan(final_data)] = -999
    final_data[np.isinf(final_data)] = -999

    # Same header as original
    out_hdr = hdulist[0].header

    return final_data, out_hdr

def interpolate_data_to_hp_galactic(data, data_hdr):    

    # Planck file in galactic coordinates
    planck_root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    Pfile = planck_root + "HFI_SkyMap_353_2048_R2.02_full.fits"

    # Planck data
    hdulist = fits.open(Pfile)
    tbdata = hdulist[1].data
    hpq = tbdata.field("Q_STOKES").flatten()
    
    gwcs = wcs.WCS(data_hdr)
    xax = np.linspace(1, data_hdr["NAXIS1"], data_hdr["NAXIS1"]).reshape(data_hdr["NAXIS1"], 1)
    yax = np.linspace(1, data_hdr["NAXIS2"], data_hdr["NAXIS2"]).reshape(1, data_hdr["NAXIS2"])
    test = gwcs.all_pix2world(xax, yax, 1)
    RA = test[0]
    Dec = test[1]
    c = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, frame="icrs")

    # Reproject into galactic coordinates
    cg = c.galactic

    hppos = hp.pixelfunc.ang2pix(hp.pixelfunc.npix2nside(50331648),  np.pi/2-np.asarray(cg.b.rad), np.asarray(cg.l.rad), nest=True)
    
    # All and final positions
    flat_hppos = hppos.flatten()
    final_data = np.zeros(hpq.size).flatten() - 999

    # Q data to place
    data = ((data).T)[:, :].flatten() # this should be upside down? Yes it is! So this is what we want.

    # zip position information and input data. 
    alldata = zip(flat_hppos, data)

    # Append duplicate key entries rather than overwriting them
    grouped_data = {}
    for k, v in alldata:
        grouped_data.setdefault(k, []).append(v)

    # Average nonzero RHT data. Do not count NaN values in histogram
    for z in grouped_data.keys():
        final_data[z] = np.nansum(grouped_data[z])/np.count_nonzero(~np.isnan(grouped_data[z]))

    final_data[np.isnan(final_data)] = -999
    final_data[np.isinf(final_data)] = -999
    
    # Same header as Planck data
    out_hdr = hdulist[0].header

    return final_data, out_hdr


# If we want to pull data from multiple velocity slices, we need to combine their RHT weights theta-bin by theta-bin.

def get_RHT_data(rht_fn):
    ipoints, jpoints, rthetas, naxis1, naxis2 = RHT_tools.get_RHT_data(rht_fn)
    npoints, nthetas = rthetas.shape
    print("There are %d theta bins" %nthetas)
    
    return ipoints, jpoints, rthetas, naxis1, naxis2, nthetas

def single_theta_slice(theta_i, ipoints, jpoints, rthetas):
    singe_theta_backprojection[jpoints, ipoints, :] = rthetas[:, theta_i]
    
    return single_theta_backprojection

# Step through theta_bins
# then step through velocities
vels = [16, 17, 18, 19, 20, 21, 22, 23, 24] # channels used in PRL
root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/"
out_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"
wlen = 75

# Get starting parameters from vels[0]
rht_fn = root + "SC_241.66_28.675.best_"+str(vels[0])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
rht_hdr = fits.getheader(rht_fn)

for theta_index in xrange(1):
    singe_theta_backprojection = np.zeros((naxis2, naxis1), np.float_)
    
    for v in xrange(vels):
        # Define RHT filename based on velocity
        rht_fn = root + "SC_241.66_28.675.best_"+str(v)+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
        
        ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints, jpoints, rthetas)
        single_theta_backprojection_galactic, out_hdr = interpolate_data_to_hp_galactic(single_theta_backprojection, rht_hdr)
        
    # Save each theta slice individually
    out_fn = "SC_241.66_28.675.best_"+str(vels[0])+"_"+str(vels[-1])+"_w"+str(wlen)+"_s15_t70_galfapixcorr_thetabin_"+str(theta_index)+".fits"        
    out_hdr["THETAI"] = theta_index
    out_hdr["VSTART"] = vels[0]
    out_hdr["VSTOP"] = vels[-1]
    
    fits.writeto(out_fn, single_theta_backprojection_galactic, out_hdr)


#out_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/Full_GALFA_mask_projected.fits"
#final_data, out_hdr = interpolate_thetas()
#fits.writeto(out_fn, final_data, out_hdr)