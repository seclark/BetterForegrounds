from __future__ import division, print_function
import numpy as np
import healpy as hp
import math
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
import copy
import time
import matplotlib.pyplot as plt
import os

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

def transform_theta_bins():
    """
    Colin's code -- stuck into this function
    """
    Nside=2048
    Npix=12*Nside**2
    thets = get_thets(75)
    thetaGal_Equ0 = hp.fitsfunc.read_map('/scr/depot1/jch/RHT_QU/rotation_maps/theta_0.0_Equ_inGal.fits')
    # # transformation check
    # for theta in thets:
    #     thetaGal = hp.fitsfunc.read_map('/scr/depot1/jch/RHT_QU/rotation_maps/theta_'+str(theta)+'_Equ_inGal.fits')
    #     for j in xrange(100): #check for first 100 pixels
    #         #print("theta_Equ = ",theta*180.0/np.pi," theta_Gal = ",thetaGal[j]*180.0/np.pi," diff = ",(theta-thetaGal[j])*180.0/np.pi)
    #         guess = thetaGal_Equ0[j] - theta
    #         if (guess < 0.0):
    #             guess += np.pi
    #         #print("theta_Equ = ",theta*180.0/np.pi," theta_Gal = ",thetaGal[j]*180.0/np.pi," guess = ",guess*180.0/np.pi,"theta_Gal-guess = ",(thetaGal[j]-guess)*180.0/np.pi)
    #         if (np.absolute((thetaGal[j]-guess)*180.0/np.pi) > 1.0e-4):
    #             print("theta_Equ = ",theta*180.0/np.pi," theta_Gal = ",thetaGal[j]*180.0/np.pi," guess = ",guess*180.0/np.pi,"theta_Gal-guess = ",(thetaGal[j]-guess)*180.0/np.pi)
    #             #print "pix=",j
    #             quit()
    # quit()

    # transformation -- has been checked on first 100 pixels of healpix map, should be OK
    thets_EquinGal = np.zeros((Npix,len(thets)))
    for i in xrange(Npix):
        for j in xrange(len(thets)):
            thets_EquinGal[i,j] = thetaGal_Equ0[i] - thets[j]
            if (thets_EquinGal[i,j] < 0.0):
                thets_EquinGal[i,j] += np.pi
    quit()


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
    
def NHI_masks():
    coldensname = "GALFA-HI_NHImap_SRcorr_VLSR-090+0090kms"
    coldensmap = fits.getdata("/Volumes/DataDavy/GALFA/DR2/NHIMaps/"+coldensname+".fits")
    coldensmap_hdr = fits.getheader("/Volumes/DataDavy/GALFA/DR2/NHIMaps/"+coldensname+".fits")
    
    nhicuts = [30, 50, 70]
    
    for n in nhicuts:
        nhipercentile = np.nanpercentile(coldensmap, n)
        cutmap = np.zeros(coldensmap.shape, np.float_)
        cutmap[np.where(coldensmap > nhipercentile)] = 1
        
        proj_data, proj_hdr = interpolate_data_to_hp_galactic(cutmap, coldensmap_hdr)
        fits.writeto("/Volumes/DataDavy/GALFA/DR2/NHIMaps/"+coldensname+"_NHI_over_percentile_{}.fits".format(n), proj_data)

def lensing_maps(local=False):
    if local is True:
        root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/thetarht_maps/"
    else:
        root = "/disks/jansky/a/users/goldston/susan/Wide_maps/QUmaps/"
    Qmap_fn = "QRHT_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70.fits"
    Umap_fn = "URHT_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70.fits"
    Imap_fn = "intrht_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70.fits"
    
    Qmap = fits.getdata(root + Qmap_fn)
    #Qmap=hp.fitsfunc.read_map(root + Qmap_fn)
    Umap = fits.getdata(root + Umap_fn)
    Imap = fits.getdata(root + Imap_fn)
    
    #coldensname = "GALFA-HI_NHImap_SRcorr_VLSR-090+0090kms"
    coldensname = "GALFA-HI_VLSR-036+0037kms_NHImap_noTcut"
    if local is True:
        nhiroot = "/Volumes/DataDavy/GALFA/DR2/NHIMaps/"
    else:
        nhiroot = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps/"
        
    coldensmap_hdr = fits.getheader(nhiroot+coldensname+".fits")
    
    hp_Q, hp_hdr = interpolate_data_to_hp_galactic(Qmap, coldensmap_hdr, local=False)
    fits.writeto(root + "DR2_allsky_Q_RHT_hp_w75_s15_t70.fits", hp_Q, hp_hdr)
    hp_U, hp_hdr = interpolate_data_to_hp_galactic(Umap, coldensmap_hdr, local=False)
    fits.writeto(root + "DR2_allsky_U_RHT_hp_w75_s15_t70.fits", hp_U, hp_hdr)
    hp_T, hp_hdr = interpolate_data_to_hp_galactic(Imap, coldensmap_hdr, local=False)
    fits.writeto(root + "DR2_allsky_int_RHT_hp_w75_s15_t70.fits", hp_T, hp_hdr)
    
    print(len(hp_Q))
    
    TQU = np.zeros((len(hp_Q), 3), np.float_)
    TQU[:, 0] = hp_T
    TQU[:, 1] = hp_Q
    TQU[:, 2] = hp_U
    
    hp_hdr['HISTORY'] = 'TQU RHT data made by Susan Clark, 2016'
    
    fits.writeto(root + "DR2_allsky_TQU_hp_w75_s15_t70.fits", TQU, hp_hdr)
    

def interpolate_data_to_hp_galactic(data, data_hdr, local=True):    

    # Planck file in galactic coordinates -- NOTE these are Nested
    #planck_root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    if local is True:
        planck_root = "/Volumes/DataDavy/Planck/"
    else:
        planck_root = "/disks/jansky/a/users/goldston/susan/Planck/"
    Pfile = planck_root + "HFI_SkyMap_353_2048_R2.02_full.fits"

    # Planck data
    hdulist = fits.open(Pfile)
    tbdata = hdulist[1].data
    hpq = tbdata.field("Q_STOKES").flatten()
    
    gwcs = wcs.WCS(data_hdr)
    xax = np.linspace(1, data_hdr["NAXIS1"], data_hdr["NAXIS1"]).reshape(data_hdr["NAXIS1"], 1)
    yax = np.linspace(1, data_hdr["NAXIS2"], data_hdr["NAXIS2"]).reshape(1, data_hdr["NAXIS2"])
    #test = gwcs.all_pix2world(xax, yax, 1, 1)
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

def single_theta_slice(theta_i, ipoints, jpoints, rthetas, naxis1, naxis2):
    single_theta_backprojection = np.zeros((naxis2, naxis1), np.float_)
    single_theta_backprojection[jpoints, ipoints] = rthetas[:, theta_i]
    
    return single_theta_backprojection
    
def reproject_by_thetabin():
    # Step through theta_bins
    # then step through velocities
    vels = [16, 17, 18, 19, 20, 21, 22, 23, 24] # channels used in PRL
    root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/"
    out_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"
    wlen = 75

    # Get starting parameters from vels[0]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[0])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints16, jpoints16, rthetas16, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
    #naxis2 = 1150
    #naxis1 = 5600
    
    # And from vels[1]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[1])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints17, jpoints17, rthetas17, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
    
    # And from vels[2]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[2])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints18, jpoints18, rthetas18, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[3]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[3])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints19, jpoints19, rthetas19, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[4]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[4])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints20, jpoints20, rthetas20, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[5]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[5])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints21, jpoints21, rthetas21, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[6]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[6])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints22, jpoints22, rthetas22, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[7]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[7])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints23, jpoints23, rthetas23, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

     # And from vels[8]
    rht_fn = root + "SC_241.66_28.675.best_"+str(vels[8])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints24, jpoints24, rthetas24, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # Original Galfa data
    galfa_fn = "/Volumes/DataDavy/GALFA/SC_241/cleaned/SC_241.66_28.675.best_20.fits"
    galfa_hdr = fits.getheader(galfa_fn)

    for theta_index in xrange(nthetas):
        time0 = time.time()
        
        #single_theta_backprojection = np.zeros((naxis2, naxis1), np.float_)
        # Might as well initialize from first vels -- that's one less velocity slice to load in each time
        single_theta_backprojection = single_theta_slice(theta_index, ipoints16, jpoints16, rthetas16, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints17, jpoints17, rthetas17, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints18, jpoints18, rthetas18, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints19, jpoints19, rthetas19, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints20, jpoints20, rthetas20, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints21, jpoints21, rthetas21, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints22, jpoints22, rthetas22, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints23, jpoints23, rthetas23, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints24, jpoints24, rthetas24, naxis1, naxis2)
        
        # Only step through other vels
        #for v in vels[1:]:
        #    # Define RHT filename based on velocity
        #    rht_fn = root + "SC_241.66_28.675.best_"+str(v)+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
        # 
        #    ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
        #    single_theta_backprojection += single_theta_slice(theta_index, ipoints, jpoints, rthetas, naxis1, naxis2)
        
        single_theta_backprojection_galactic, out_hdr = interpolate_data_to_hp_galactic(single_theta_backprojection, galfa_hdr)
        time1 = time.time()
        print("theta %f took %f minutes" %(theta_index, (time1 - time0)/60.))
    
        # Save each theta slice individually
        out_fn = out_root + "SC_241.66_28.675.best_"+str(vels[0])+"_"+str(vels[-1])+"_w"+str(wlen)+"_s15_t70_galfapixcorr_thetabin_"+str(theta_index)+".fits"        
        out_hdr["THETAI"] = theta_index
        out_hdr["VSTART"] = vels[0]
        out_hdr["VSTOP"] = vels[-1]
    
        fits.writeto(out_fn, single_theta_backprojection_galactic, out_hdr)
        
def get_extra0_sstring(cstart, cstop):
    """
    For naming convention
    """
    if cstart <= 999:
        s_string = "S0"
        extra_0 = "0"
    else:
        s_string = "S"
        extra_0 = ""
    if cstart == 999:
        s_string = "S0"
        extra_0 = ""
        
    return s_string, extra_0
    
def get_extra0_startstop(cstart, cstop):
    """
    For naming convention
    """
    if cstart <= 999:
        start_0 = "0"
        end_0 = "0"
    else:
        start_0 = ""
        end_0 = ""
    if cstart == 999:
        start_0 = "0"
        end_0 = ""
        
    return start_0, end_0
    
def single_thetabin_single_vel_allsky(velnum=-8):

    wlen = 75
    cstep = 5 

    # Everything is in chunks of 5 channels. e.g. 1024_1028 includes [1024, 1028] inclusive.
    cstart = 1024 + velnum*cstep
    cstop = cstart + cstep - 1
    s_string, extra_0 = get_extra0_sstring(cstart, cstop)
    
    velrangestring = s_string+str(cstart)+"_"+extra_0+str(cstop)

    root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
    out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velrangestring+"/"
    
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    
    # Overlapping/stepping parameters
    step = 3600
    filler_overlap = 60 #50
    normal_overlap = 50
    leftstop = 111
    rightstart = 21488
    
    # Shape of the all-sky data
    nyfull = 2432
    nxfull = 21600
    fulldata = np.zeros((nyfull, nxfull), np.float_)
    
    # Loop through thetas - should be xrange(ntheta) but just testing now
    for theta_index in np.arange(0, 165):#xrange(166):#xrange(1):
        time0 = time.time()
        
        # New single theta backprojection
        fulldata = np.zeros(fulldata.shape)
    
        for num in [0, 1, 2, 3, 4, 5]:
            
            # get normal start/stop
            xstart0_normal = max((step*num - normal_overlap), 0)
            xstop0_normal = step*(num + 1) + normal_overlap

            # Load normal xyt data
            if num == 3:
                rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_"+str(num)+"_SRcorr_fakecrpix1_xyt_w"+str(wlen)+"_s15_t70.fits"
            else:
                rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_"+str(num)+"_SRcorr_xyt_w"+str(wlen)+"_s15_t70.fits"
            ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
            single_theta_backprojection_chunk = single_theta_slice(theta_index, ipoints, jpoints, rthetas, naxis1, naxis2)
            
            fulldata = place_normal_data(fulldata, single_theta_backprojection_chunk, xstart0_normal, xstop0_normal)

            # Load filler xyt data
            if num > 0:
                # get filler start/stop
                #xstart0_filler, xstop0_filler = get_start_stop_from_fillernum(num, filler_overlap)
                
                if num == 3:
                    rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_filler"+str(num)+"_SRcorr_fakecrpix1_xyt_w"+str(wlen)+"_s15_t70.fits"
                else:
                    rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_filler"+str(num)+"_SRcorr_xyt_w"+str(wlen)+"_s15_t70.fits"
                ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
                single_theta_backprojection_chunk = single_theta_slice(theta_index, ipoints, jpoints, rthetas, naxis1, naxis2)
                
                #fulldata = place_filler_data(fulldata, single_theta_backprojection_chunk, xstart0_filler, xstop0_filler)
                fulldata = place_filler_data(fulldata, single_theta_backprojection_chunk, num, filler_overlap)
            
        # Load seam xyt data
        rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_seam_SRcorr_fakecrpix1_xyt_w"+str(wlen)+"_s15_t70.fits"
        ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
        single_theta_backprojection_chunk = single_theta_slice(theta_index, ipoints, jpoints, rthetas, naxis1, naxis2)
        
        fulldata = place_seam_data(fulldata, single_theta_backprojection_chunk, leftstop, rightstart)
            
        hdr = fits.getheader("/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRcorr/GALFA-HI_NHI_VLSR-90+90kms/data/GALFA-HI_NHI_VLSR-90+90kms.fits")
        hdr['VMIN'] = cstart
        hdr['VMAX'] = cstop
        hdr['THET'] = theta_index
        fits.writeto(out_root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_SRcorr_w"+str(wlen)+"_s15_t70_theta_"+str(theta_index)+".fits", fulldata, hdr)

        time1 = time.time()
        print(np.nansum(fulldata))
        print("theta %f took %f minutes" %(theta_index, (time1 - time0)/60.))
            
def place_normal_data(holey_data, filler_data, xstart0, xstop0):
    #holey_data[:, xstart0:xstop0] = filler_data
    
    filler_data[np.where(holey_data[:, xstart0:xstop0] != 0)] = 0
    holey_data[:, xstart0:xstop0] += filler_data
    
    return holey_data
            
def place_filler_data(holey_data, filler_data, fillernum, overlap):

    # Add filler, blanking regions that already contain data. THIS IS WRONG - creates small hole
    #filler_data[np.where(holey_data[:, xstart0:xstop0] != 0)] = 0
    #holey_data[:, xstart0:xstop0] += filler_data
    
    # instead, just replace rather than add. THIS IS WRONG because edges are 0
    #holey_data[:, xstart0:xstop0] = filler_data
    
    infiller0, infiller1, tofiller0, tofiller1 = get_placement_from_fillernum(fillernum, overlap)
    holey_data[:, tofiller0:tofiller1] = filler_data[:, infiller0:infiller1]
    #print('placing data from {} to {}'.format(tofiller0, tofiller1))
    
    return holey_data
    
def place_seam_data(holey_data, seam_data, leftstop, rightstart):
    
    # Blank seam regions that already contain data
    ny_left, nx_left = holey_data[:, :leftstop].shape
    ny_right, nx_right = holey_data[:, rightstart:].shape
    
    left_seam_data = seam_data[:, :nx_right]
    right_seam_data = seam_data[:, nx_right:] 
    
    left_seam_data[np.where(holey_data[:, rightstart:] != 0)] = 0
    right_seam_data[np.where(holey_data[:, :leftstop] != 0)] = 0
    
    seam_data[np.where(np.isnan(seam_data) == True)] = 0
    
    holey_data[:, :leftstop] += right_seam_data
    holey_data[:, rightstart:] += left_seam_data
    
    return holey_data
    
def get_start_stop_from_fillernum(fillernum, overlap):
    
    if fillernum == 1:
        xstart0 = 3598 - overlap
        xstop0 = 3601 + overlap
        
    if fillernum == 2:
        xstart0 = 7198 - overlap
        xstop0 = 7201 + overlap
    
    if fillernum == 3:
        xstart0 = 10798 - overlap
        xstop0 = 10801 + overlap
    
    if fillernum == 4:
        xstart0 = 14398 - overlap
        xstop0 = 14401 + overlap
        
    if fillernum == 5:
        xstart0 = 17998 - overlap
        xstop0 = 18001 + overlap
        
    return xstart0, xstop0
    
def get_placement_from_fillernum(fillernum, overlap):
    xstart0, xstop0 = get_start_stop_from_fillernum(fillernum, overlap)
    infiller0 = overlap - 2
    infiller1 = xstop0 - overlap + 2 - xstart0
    tofiller0 = xstart0 + infiller0
    tofiller1 = xstart0 + infiller1
    
    return infiller0, infiller1, tofiller0, tofiller1
    
    
def redo_local_intrhts(velnum=-10):
    
    wlen = 75
    cstep = 5 
    filler_overlap = 60
    leftstop = 111 # seam info
    rightstart = 21488

    # Everything is in chunks of 5 channels. e.g. 1024_1028 includes [1024, 1028] inclusive.
    cstart = 1024 + velnum*cstep
    cstop = cstart + cstep - 1
    start_0, end_0 = get_extra0_startstop(cstart, cstop)
    
    # vel range string when no leading S
    velrangestring = start_0+str(cstart)+"_"+end_0+str(cstop)
    
    root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/thetarht_maps/"
    
    holey_intrht = np.load(root + "intrht_allsky_ch"+velrangestring+"_SRcorr_w75_s15_t70.npy")
    
    for num in [1, 2, 3, 4, 5]:
        
        filler_intrht = np.load(root + "intrht_allsky_ch"+velrangestring+"_filler"+str(num)+"_SRcorr_w75_s15_t70.npy")
        holey_intrht = place_filler_data(holey_intrht, filler_intrht, num, filler_overlap)
        
    seam_intrht = np.load(root + "intrht_allsky_ch"+velrangestring+"_seam_SRcorr_w75_s15_t70.npy")
    final_intrht = place_seam_data(holey_intrht, seam_intrht, leftstop, rightstart)
    
    np.save(root + "intrht_filled_allsky_ch"+velrangestring+"_SRcorr_w75_s15_t70.npy", final_intrht)
    
        
def reproject_by_thetabin_allsky():
    """
    Reproject single theta backprojections for full GALFA sky, incl. wrapped and filled data.
    Note: working code for this developed at Berkeley because the I/O requirement is too much.
    """
    
    vels = np.arange(-10, 11, 1) # channels analyzed in full-sky data
    root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/xyt_data/" 
    out_root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/xyt_data/"
    wlen = 75
    cstep = 5
    
    # Overlapping/stepping parameters
    step = 3600
    overlap = 50
    
    # This is just to get the shape of the all-sky data
    fulldata = fits.getdata("/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/GALFA_HI_W_S1024_1028.fits")
    nyfull, nxfull = fulldata.shape
    
    # Loop through thetas - should be xrange(ntheta) but just testing now
    for theta_index in xrange(1):
        time0 = time.time()
        
        # New single theta backprojection
        single_theta_backprojection = np.zeros(fulldata.shape)
        
        # Step through velocity channels
        for v_ in vels: # Everything is in chunks of 5 channels. e.g. 1024_1028 includes [1024, 1028] inclusive.
            cstart = 1024 + v_*cstep
            cstop = cstart + cstep - 1
        
            s_string, extra_0 = get_extra0_sstring(cstart, cstop)
    
            for num in [0, 1, 2, 3, 4, 5]:
                time2 = time.time()
                # Start and stop for each section
                xstart0 = max((step*num - overlap), 0)
                xstop0 = step*(num + 1) + overlap
        
                # Load xyt data
                rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_"+str(num)+"_SRcorr_xyt_w"+str(wlen)+"_s15_t70.fits"
                ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
                single_theta_backprojection_chunk = single_theta_slice(theta_index, ipoints, jpoints, rthetas, naxis1, naxis2)
              
                single_theta_backprojection_chunk[np.where(np.isnan(single_theta_backprojection_chunk) == True)] = 0   
                single_theta_backprojection[:, xstart0:xstop0] += single_theta_backprojection_chunk
                
                # While we're here, also load associated filler
                #rht_fn = root+"GALFA_HI_W_"+s_string+str(cstart)+"_"+extra_0+str(cstop)+"_newhdr_filler"+str(num)+"_SRcorr_xyt_w"+str(wlen)+"_s15_t70.fits"
                #ipoints, jpoints, rthetas, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
                
                time3 = time.time()
                print("num %f took %f minutes" %(num, (time3 - time2)/60.))
        
        time1 = time.time()
        print("theta %f took %f minutes" %(theta_index, (time1 - time0)/60.))
        
    return single_theta_backprojection

def plot_by_thetabin():

    unprojected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/"
    unprojected_fn = "SC_241.66_28.675.best_24_xyt_w75_s15_t70_galfapixcorr.fits"
    
    vels = [16, 17, 18, 19, 20, 21, 22, 23, 24] # channels used in PRL
    wlen = 75

    # Get starting parameters from vels[0]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[0])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints16, jpoints16, rthetas16, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
    
    # And from vels[1]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[1])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints17, jpoints17, rthetas17, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
    
    # And from vels[2]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[2])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints18, jpoints18, rthetas18, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[3]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[3])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints19, jpoints19, rthetas19, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[4]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[4])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints20, jpoints20, rthetas20, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[5]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[5])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints21, jpoints21, rthetas21, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[6]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[6])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints22, jpoints22, rthetas22, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

    # And from vels[7]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[7])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints23, jpoints23, rthetas23, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)

     # And from vels[8]
    rht_fn = unprojected_root + "SC_241.66_28.675.best_"+str(vels[8])+"_xyt_w"+str(wlen)+"_s15_t70_galfapixcorr.fits"
    ipoints24, jpoints24, rthetas24, naxis1, naxis2, nthetas = get_RHT_data(rht_fn)
    
    
    for theta_index in xrange(1):
        time0 = time.time()

        # Might as well initialize from first vels -- that's one less velocity slice to load in each time
        single_theta_backprojection = single_theta_slice(theta_index, ipoints16, jpoints16, rthetas16, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints17, jpoints17, rthetas17, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints18, jpoints18, rthetas18, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints19, jpoints19, rthetas19, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints20, jpoints20, rthetas20, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints21, jpoints21, rthetas21, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints22, jpoints22, rthetas22, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints23, jpoints23, rthetas23, naxis1, naxis2)
        single_theta_backprojection += single_theta_slice(theta_index, ipoints24, jpoints24, rthetas24, naxis1, naxis2)        
    
    projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"
    projected_fn = "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_99.fits"
    
    cmap = "bone_r"
    projected_data = fits.getdata(projected_root + projected_fn)
    plot_projected = hp.mollview(np.clip(projected_data, 0, np.nanmax(projected_data)), return_projected_map = True, nest = True)
    
    fig = plt.figure(figsize = (6, 6.5))
    ax1 = fig.add_subplot(211)
    
    ny, nx = plot_projected.shape
    im1 = ax1.imshow(np.log10(plot_projected), cmap = cmap)
    plt.colorbar(im1)
    #ax1.set_ylim(0, ny)
    ax1.set_ylim(200, 400)
    ax1.set_xlim(200, 500)
    ax2 = fig.add_subplot(212)
    im2 = ax2.imshow(np.log10(single_theta_backprojection), cmap = cmap)
    plt.colorbar(im2, orientation = "horizontal")

if __name__ == "__main__":
#    plot_by_thetabin()
     #lensing_maps()
     single_thetabin_single_vel_allsky(velnum=-10) # running: -10, -9, -8
     
     #redo_local_intrhts(velnum=-9)
    