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


def make_single_theta_TQU_map(thetabin):
    # TQU map of Equatorial data
    
    TQUmap = np.zeros((3,Npix))
    TQUmap[0][:] = 1

    # let's do a test: thets[1] = 0.019039955476301777. In RHT angle space, aka Equ, IAU B-field angle.
    thets1 = thets_unproj[thetabin]
    thets1U = np.sin(2*thets1)
    thets1Q = np.cos(2*thets1)
    #TQUmap[0][:] = 1
    TQUmap[1][:] = -thets1Q # convert to Planck dust pol angle
    TQUmap[2][:] = thets1U

    # change to RING ordered
    print('changing ordering to RING')
    for _tqu in range(3):
        TQUmap[_tqu, :] = hp.reorder(TQUmap[_tqu, :], n2r=True)
        
    return TQUmap
    
def make_test_TQU_map(place_id, unprojected_theta):
    TQUmap = np.zeros((3,Npix))
    #TQUmap[0][place_id] = 1
    TQUmap[0][:] = 1
    thets1U = np.sin(2*unprojected_theta)
    thets1Q = np.cos(2*unprojected_theta)
    #TQUmap[1][place_id] = -thets1Q # convert to Planck dust pol angle
    #TQUmap[2][place_id] = thets1U
    TQUmap[1][:] = -thets1Q # convert to Planck dust pol angle
    TQUmap[2][:] = thets1U


    # change to RING ordered
    print('changing ordering to RING')
    for _tqu in range(3):
        TQUmap[_tqu, :] = hp.reorder(TQUmap[_tqu, :], n2r=True)
        
    return TQUmap

    
def rotate_TQU_Equ_to_Gal(TQUmap, plot=False):
    out_root = '../data/'

    # make placeholder TQU map - this is RING ordered, Equatorial angle, Galactic coordinates
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
    
    # remove temp files
    call("ls /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)
    call("rm /Users/susanclark/BetterForegrounds/data/temp*.fits", shell=True, stdout=PIPE)
    
    if plot:
        # plot theta map
        thetaGal = np.mod(0.5*np.arctan2(TQUmapGal[2], TQUmapGal[1]), np.pi)
        hp.mollview(thetaGal, unit='rad', title='theta_RHT_Equ_inGal', min=0.0, max=np.pi, coord='G')
        
    return TQUmapGal


thets_unproj = rht_to_planck.get_thets(75)

#for _i in xrange(165):  
#    TQUmapEqu = make_single_theta_TQU_map(_i)
#    TQUmapGal = rotate_TQU_Equ_to_Gal(TQUmapEqu, plot=False)
#    
#    final_out_root = "/Volumes/DataDavy/Foregrounds/coords/thetabin_rotation_maps/"
#    hp.fitsfunc.write_map(final_out_root+'hp_projected_thetabin_'+str(_i)+'.fits', TQUmapGal, coord='G')
#    
#    print('thetabin {} complete'.format(_i))

unprojected_theta = 2.9307297740993126 #0.62219971868954638
place_id = 18691216 
TQUmapEqu = make_test_TQU_map(place_id, unprojected_theta)  
TQUmapGal = rotate_TQU_Equ_to_Gal(TQUmapEqu, plot=False)
final_out_root = "/Volumes/DataDavy/Foregrounds/coords/thetabin_rotation_maps/"
hp.fitsfunc.write_map(final_out_root+'hp_projected_test_pix_'+str(place_id)+'_2.9_all.fits', TQUmapGal, coord='G')

    