import numpy as np
import matplotlib
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
import subprocess
import os
"""
Code to take a boolean mask and apply an apodization window that smoothly interpolates from 0 to 1
Uses the fortran healpix routine process_mask
User needs to supply boolean mask file and define various parameters
"""

#####
# your copy of the healpix process_mask routine
process_mask_routine = '/home/jch/Healpix_3.31/bin_gfortran/process_mask'
#####
# define a gaussian tapering function with FWHM = FWHM_apod; the function asymptotes to unity far from invalid pixels and zero at invalid pixels
def taper_func(dist_rad, sigma_apod): #dist_rad = angular distance in rad, sigma_apod = std dev of gaussian apodization kernel
    return 1.-np.exp(-dist_rad**2./(2.*sigma_apod**2.))
####
# definitions
FITS_end = '.fits'
TXT_end = '.txt'
PDF_end = '.pdf'
# we will not fill any holes in the mask in this routine!
hole_min_size = '0'
hole_min_surf_arcmin2 = '0.'
filled_file = '\'\'' #we don't care about this since we're not filling in any holes in the mask
####


#####
# parameters of the mask map
# mask_dir = '/data/jch/RHT_QU/'
# mask_name = 'allsky_GALFA_mask_nonzero_nchannels_edge200_hp_proj_plusbgt70_maskGal'
# Nside=2048
# coords = 'G' #coordinate system
# FWHM_apod_arcmin = 15. #FWHM of the gaussian apodization function
# PLOT_OPT = option whether to plot images of masks
def apodize_mask(mask_dir, mask_name, Nside, coords, FWHM_apod_arcmin, PLOT_OPT=False):
    # input/parameters for process_mask Healpix routine
    Npix = 12*Nside**2
    mask_file = mask_dir+mask_name+FITS_end
    distance_file = mask_dir+mask_name+'_dist'+FITS_end
    # apodization parameters (for the tapering function)
    FWHM_apod = FWHM_apod_arcmin*1./60.*np.pi/180. #convert to rad
    sigma_apod = FWHM_apod/(2.*np.sqrt(2.*np.log(2.))) #convert from FWHM to sigma
    #####
    # construct the distance map
    # check if the distance_file has already been created for some reason
    DIST_FILE_EXISTS = False
    if os.path.exists(distance_file):
        DIST_FILE_EXISTS = True # file exists
        # if distance file doesn't exist, construct it
    if (DIST_FILE_EXISTS == False):
        # temporarily create the parameters file that will be needed for process_mask
        f = open('process_mask_params.txt','w')
        f.write('mask_file = '+mask_file+'\n')
        f.write('hole_min_size = '+hole_min_size+'\n')
        f.write('hole_min_surf_arcmin2 = '+hole_min_surf_arcmin2+'\n')
        f.write('filled_file = '+filled_file+'\n')
        f.write('distance_file = '+distance_file+'\n')
        f.close()
        # call process_mask to get the distance map
        subprocess.call([process_mask_routine, 'process_mask_params.txt'])
        # remove the temporary parameters file
        subprocess.call(['rm', '-f', 'process_mask_params.txt'])
    #####
    # apply apodization
    # read in boolean mask and distance map
    mask = hp.read_map(mask_file, verbose=False)
    mask_dist = hp.read_map(distance_file, verbose=False)
    # apply apodization to the mask
    taper_map = taper_func(mask_dist,sigma_apod)
    mask_apod = mask.astype(np.float)*taper_map
    # images, if desired
    if (PLOT_OPT == True):
        plt.clf()
        hp.mollview(mask_apod, coord=coords, min=0., max=1., title='apodized mask')
        plt.savefig(mask_dir+mask_name+PDF_end)
    # compute and save fsky and second moment
    fsky_apod = np.sum(mask_apod)/float(Npix)
    fsky_apod2 = np.sum(mask_apod**2.)/float(Npix)
    np.savetxt(mask_dir+mask_name+'_fsky'+TXT_end, np.transpose(np.array([fsky_apod,fsky_apod2])))
    # save apodized mask
    hp.write_map(mask_dir+mask_name+'_taperFWHM'+str(int(FWHM_apod_arcmin))+'arcmin'+FITS_end, mask_apod, coord=coords, overwrite=True)
    #####
