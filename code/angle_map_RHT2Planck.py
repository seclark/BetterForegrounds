import numpy as np
import glob, pickle
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
from subprocess import call, PIPE

Nside=2048
Npix=12*Nside**2

# read in RHT angle bin values
thets = np.loadtxt('RHTthetabins_wlen75.txt')

## code to rename png files for use in animated gif
#for i in xrange(len(thets)):
#    thetaGal = hp.fitsfunc.read_map('/scr/depot1/jch/RHT_QU/rotation_maps/theta_'+str(thets[i])+'_Equ_inGal.fits')
#    plt.clf()
#    hp.mollview(thetaGal, unit='rad', title='theta_'+str(thets[i])+'_Equ_inGal', min=0.0, max=np.pi, coord='G')
#    filename = "/scr/depot1/jch/RHT_QU/rotation_maps/numbered/theta_{:0>3d}_Equ_inGal.png".format(i)
#    plt.savefig(filename)
#quit()
    

# construct map of each theta value
# - convert to QU and translate from IAU + B-field to Planck + dust pol
# - make placeholder TQU map
# - convert the TQU map to alm^TEB using anafast
# - rotate the alm^TEB from Equ to Gal coords using alteralm
# - convert the rotated alm^TEB back to a real-space TQU map in new coords
# - compute theta at each pixel
# - save the resulting map, so we'll have them for later use in making interpolating function
for theta in thets:
    plt.clf()
    hp.mollview(theta*np.ones(Npix), unit='rad', title='theta_'+str(theta)+'_Equ', min=0.0, max=np.pi, coord='C')
    plt.savefig('/scr/depot1/jch/RHT_QU/rotation_maps/theta_'+str(theta)+'_Equ.png')
    TQUmap = np.zeros((3,Npix))
    TQUmap[1] = -np.cos(2.0*theta) # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT 
    TQUmap[2] = np.sin(2.0*theta) # convert from "IAU B-field angle" to "Planck/Healpix dust polarization angle": U_RHT -> U_RHT, Q_RHT -> -Q_RHT
    hp.fitsfunc.write_map('/scr/depot1/jch/RHT_QU/rotation_maps/temp.fits', TQUmap, coord='C') #have to save map to use with f90 healpix utilities
    call("/u/jch/Healpix_2.20a/binf90/anafast anafast_paramfile.txt", shell=True, stdout=PIPE)
    call("/u/jch/Healpix_2.20a/binf90/alteralm alteralm_paramfile.txt", shell=True, stdout=PIPE)
    call("/u/jch/Healpix_2.20a/binf90/synfast synfast_paramfile.txt", shell=True, stdout=PIPE)
    TQUmapGal = np.zeros((3,Npix))
    TQUmapGal[0], TQUmapGal[1], TQUmapGal[2] = hp.fitsfunc.read_map('/scr/depot1/jch/RHT_QU/rotation_maps/temp_Gal.fits', field=(0,1,2))
    thetaGal = np.mod(0.5*np.arctan2(TQUmapGal[2], TQUmapGal[1]), np.pi)
    hp.fitsfunc.write_map('/scr/depot1/jch/RHT_QU/rotation_maps/theta_'+str(theta)+'_Equ_inGal.fits', thetaGal, coord='G')
    call("rm /scr/depot1/jch/RHT_QU/rotation_maps/temp*.fits", shell=True, stdout=PIPE)
    plt.clf()
    hp.mollview(thetaGal, unit='rad', title='theta_'+str(theta)+'_Equ_inGal', min=0.0, max=np.pi, coord='G')
    plt.savefig('/scr/depot1/jch/RHT_QU/rotation_maps/theta_'+str(theta)+'_Equ_inGal.png')
    #quit()
