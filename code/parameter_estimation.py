from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
from astropy.io import fits
import cPickle as pickle
import itertools
import string
import sqlite3
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import matplotlib.ticker as ticker
import copy
import os

# Local repo imports
import debias
import rht_to_planck

# Other repo imports (RHT helper code)
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

import galfa_name_lookup

"""
 Simple psi, p estimation routines.
"""

def plot_p(tp):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax_posterior(ax1, tp.sample_p0, tp.sample_psi0, tp.normed_posterior)

def ax_posterior(ax, p0s, psi0s, B2D, cmap = "hsv", colorbar = False):
    """
    Plot 2D posterior distribution on given axes instance
    """
    
    obj = ax.pcolor(p0s, psi0s, B2D, cmap = cmap)
    #obj = ax.pcolormesh(p0s, psi0s, B2D, cmap = cmap)
    maxval = np.nanmax(B2D)
    levels = np.asarray([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9])*maxval
    obj = ax.contour(p0s, psi0s, B2D, levels, extend = "both", colors = "gray")
    ax.set_xlim(np.nanmin(p0s), np.nanmax(p0s))
    ax.set_ylim(np.nanmin(psi0s), np.nanmax(psi0s))
    ax.set_xlabel(r"$p_0$", size = 15)
    ax.set_ylabel(r"$\psi_0$", size = 15)
    if colorbar == True:
        plt.colorbar(obj)
   
def plot_randomsample_posteriors(pmeas, psimeas, p0s, psi0s, posteriors, cmap = "hsv", overplotmeas = True, sigpGsq = None):
    """
    Plot 9 randomly chosen 2D posteriors.
    """
    
    nposts, lpostssq = posteriors.shape
    lposts = np.round(np.sqrt(lpostssq))
    
    fig = plt.figure(figsize = (12, 10), facecolor = "white")
    for i in xrange(9):
        indx = np.random.randint(nposts)
        ax = fig.add_subplot(3, 3, i)
        ax_posterior(ax, p0s, psi0s, posteriors[indx, :].reshape(lposts, lposts), cmap = cmap)
        
        # Overplot measured values
        if overplotmeas == True:
            ax.plot([pmeas[indx], pmeas[indx]], [0, np.pi], '--', color = "pink", lw = 1)
            ax.plot([0, 1], [psimeas[indx], psimeas[indx]], '--', color = "pink", lw = 1)
            
        if sigpGsq != None:
            ax.set_title(r"$"+str(indx)+",$ $p_{meas}/\sigma_{p, G} = "+str(np.round(pmeas[indx]/np.sqrt(sigpGsq[indx]), 1))+"$", size = 15)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    
def plot_test_posteriors(pmeas, psimeas, p0s, psi0s, posteriors, cmap = "hsv", overplotmeas = True, sigpGsq = None, rollax = True):
    """
    Plot the 20 test 2D posteriors.
    """
    
    nposts, lpostssq = posteriors.shape
    lposts = np.round(np.sqrt(lpostssq))
    
    psi0s_rolled = psi0s - np.pi/2.0
    
    fig = plt.figure(figsize = (12, 10), facecolor = "white")
    for i in xrange(20):
        ax = fig.add_subplot(4, 5, i+1)
        post = posteriors[i, :].reshape(lposts, lposts)
        if rollax == True:
            post = np.roll(post, np.int(lposts/2.0), axis=0)
            psi0s = psi0s_rolled
        ax_posterior(ax, p0s, psi0s, post, cmap = cmap)
        
        # Overplot measured values
        if overplotmeas == True:
            if rollax == True:
                ax.plot([pmeas[i], pmeas[i]], [-np.pi/2.0, np.pi/2.0], '--', color = "pink", lw = 1)
            else:                
                ax.plot([pmeas[i], pmeas[i]], [0, np.pi], '--', color = "pink", lw = 1)
            ax.plot([0, 1], [psimeas[i], psimeas[i]], '--', color = "pink", lw = 1)
            
        if sigpGsq != None:
            ax.set_title(r"$p_{meas}/\sigma_{p, G} = "+str(np.round(pmeas[i]/sigpGsq[i], 1))+"$", size = 15)
        
        # Change p0 limits for high SNR cases
        if i >= 10:
            ax.set_xlim(0, 0.8)
        if i >= 15:
            ax.set_xlim(0, 0.2)
        
        plt.yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$\pi/4$", r"$\pi/2$"])
        plt.tick_params(axis='both', which='major', labelsize=10)
        
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)

def eps_from_cov(covmat, twoD = False):
    """
    Take covariance matrix, returns epsilon = sigma_Q/sigma_U
    2D == True :: 2D covariance matrix; otherwise 3D is assumed (I info included)
    """
    
    if twoD == True:
        eps = np.sqrt(covmat[0, 0]/covmat[1, 1])
    else: 
        eps = np.sqrt(covmat[1, 1]/covmat[2, 2])
        
    return eps
        
def rho_from_cov(covmat, twoD = False):
    """
    Take covariance matrix, returns rho = sigma_QU /(sigma_Q sigma_U)
    2D == True :: 2D covariance matrix; otherwise 3D is assumed (I info included)
    """
    
    if twoD == True:
        rho = covmat[0, 1]/(np.sqrt(covmat[1, 1])*np.sqrt(covmat[0, 0]))
    else: 
        rho = covmat[1, 2]/(np.sqrt(covmat[2, 2])*np.sqrt(covmat[1, 1]))
        
    return rho    


def get_Planck_data(Nside = 2048):
    """
    Get TQU and covariance matrix data.
    Currently in Galactic coordinates.
    """

    full_planck_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/HFI_SkyMap_353_"+str(Nside)+"_R2.02_full.fits"

    # resolution
    Npix = 12*Nside**2

    # input Planck 353 GHz maps (Galactic)
    # full-mission -- N.B. these maps are already in RING ordering, despite what the header says
    map353Gal = np.zeros((3,Npix)) #T,Q,U
    cov353Gal = np.zeros((3,3,Npix)) #TT,TQ,TU,QQ,QU,UU
    map353Gal[0], map353Gal[1], map353Gal[2], cov353Gal[0,0], cov353Gal[0,1], cov353Gal[0,2], cov353Gal[1,1], cov353Gal[1,2], cov353Gal[2,2], header353Gal = hp.fitsfunc.read_map(full_planck_fn, field=(0,1,2,4,5,6,7,8,9), h=True)

    return map353Gal, cov353Gal
    
def test_posteriors():
    """
    Calculate 2d Bayesian posteriors of test cases in Montier+ 2015 II.
    """
    # Number of covariance matrix examples
    Ncov = 5
    
    # eps = 1, rho = 0, (eps_eff = 1, theta = 0)
    eps = np.asarray([1.0, 0.5, 2.0, 1.0, 1.0])
    rho = np.asarray([0, 0, 0, -0.5, 0.5])
    
    # Set pmeas = p0 = 0.1, psimeas = psi0 = 0
    psimeas = np.repeat(0.0, Ncov)
    pmeas = np.repeat(0.1, Ncov)
    
    # For SNRs p0/sig_p,G = 0.1, 0.5, 1.0, 5.0
    snrs = np.asarray([0.1, 0.5, 1.0, 5.0])
    snrs = snrs.reshape(len(snrs), 1)
    sig_pG = (pmeas.reshape(1, Ncov)/snrs).flatten()
    
    pmeas = np.repeat(pmeas, len(snrs)).flatten()
    psimeas = np.repeat(psimeas, len(snrs)).flatten()
    eps = np.tile(eps, len(snrs)).flatten()
    rho = np.tile(rho, len(snrs)).flatten()
    snrs = np.repeat(snrs, Ncov, axis=1).flatten()
    
    # Number of total examples
    Nex = len(snrs)
    
    covmatrix = np.zeros((2, 2, Nex), np.float_)
    for i in xrange(Nex):
        covmatrix[0, 0, i] = (sig_pG[i]**2/np.sqrt(1 - rho[i]**2))*eps[i]
        covmatrix[0, 1, i] = (sig_pG[i]**2/np.sqrt(1 - rho[i]**2))*rho[i]
        covmatrix[1, 0, i] = (sig_pG[i]**2/np.sqrt(1 - rho[i]**2))*rho[i]
        covmatrix[1, 1, i] = (sig_pG[i]**2/np.sqrt(1 - rho[i]**2))*1.0/eps[i]
    
    invsig = np.linalg.inv(covmatrix.swapaxes(0, 2))
    
    print("Done inverting sigma_p")

    # Create grid of psi0's and p0's to sample
    nsample = 100
    psi0_all = np.linspace(0, np.pi, nsample)
    p0_all = np.linspace(0, 1.0, nsample)

    p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))
    p0_psi0_pairs = zip(p0_psi0_grid[0, ...].ravel(), p0_psi0_grid[1, ...].ravel())
    
    print("starting slow way")
    time0 = time.time()
    out = np.zeros((Nex, nsample*nsample), np.float_) 
    rharr = np.zeros((2, 1), np.float_)
    lharr = np.zeros((1, 2), np.float_)
    
    for p, isig in enumerate(invsig): 

        for (i, (p0, psi0)) in enumerate(p0_psi0_pairs):

            # Implement Montier+ II Eq. 24
            rharr[0, 0] = pmeas[p]*np.cos(2*psimeas[p]) - p0*np.cos(2*psi0)
            rharr[1, 0] = pmeas[p]*np.sin(2*psimeas[p]) - p0*np.sin(2*psi0)

            # Lefthand array is transpose of righthand array
            lharr = rharr.T
    
            out[p, i] = (1/(np.pi*np.sqrt(sig_pG[p])))*np.exp(-0.5*np.dot(lharr, np.dot(isig, rharr)))
    time1 = time.time()
    print("process took ", time1 - time0, "seconds")
    
    plot_test_posteriors(pmeas, psimeas, p0_all, psi0_all, out, cmap = "jet")
    
    return out, covmatrix

def Planck_posteriors(map353Gal = None, cov353Gal = None, firstnpoints = 1000, plotrandomsample = True):
    """
    Calculate 2D Bayesian posteriors for Planck data.
    """
    # resolution
    Nside = 2048
    Npix = 12*Nside**2
    if map353Gal == None:
        map353Gal, cov353Gal = get_Planck_data(Nside = Nside)

    # sigma_p as defined in arxiv:1407.0178v1 Eqn 3.
    sigma_p = np.zeros((2, 2, Npix)) # [sig_Q^2, sig_QU // sig_QU, UU]
    sigma_p[0, 0, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 1, :] #QQ
    sigma_p[0, 1, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 2, :] #QU
    sigma_p[1, 0, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 2, :] #QU
    sigma_p[1, 1, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[2, 2, :] # UU

    # Assume rho = 1, so define det(sigma_p) = sigma_p,G^4
    det_sigma_p = np.linalg.det(sigma_p.swapaxes(0, 2))
    sigpGsq = np.sqrt(det_sigma_p)

    # measured polarization angle (psi_i = arctan(U_i/Q_i))
    psimeas = np.mod(0.5*np.arctan2(map353Gal[2, :], map353Gal[1, :]), np.pi)

    # measured polarization fraction
    pmeas = np.sqrt(map353Gal[1, :]**2 + map353Gal[2, :]**2)/map353Gal[0, :]

    # temporary hack -- to only look at first n points
    if firstnpoints != None:
        Npix = firstnpoints
    sigma_p = sigma_p[:, :, 0:Npix]
    
    # invert matrix -- must have Npix axis first
    invsig = np.linalg.inv(sigma_p.swapaxes(0, 2))
    
    print("Done inverting sigma_p")

    # Create grid of psi0's and p0's to sample
    nsample = 100
    psi0_all = np.linspace(0, np.pi, nsample)
    p0_all = np.linspace(0, 1.0, nsample)

    p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))

    # Testing new "fast way" that works for isig array of size (2, 2, nsample*nsample) s.t. loop is over Npix
    print("starting fast way")
    time0 = time.time()
    outfast = np.zeros((Npix, nsample*nsample), np.float_)
    
    # These have length Npix
    measpart0 = pmeas*np.cos(2*psimeas)
    measpart1 = pmeas*np.sin(2*psimeas)
    
    p0pairs = p0_psi0_grid[0, ...].ravel()
    psi0pairs = p0_psi0_grid[1, ...].ravel()
    
    # These have length nsample*nsample
    truepart0 = p0pairs*np.cos(2*psi0pairs)
    truepart1 = p0pairs*np.sin(2*psi0pairs)
    
    rharrbig = np.zeros((2, 1, nsample*nsample), np.float_)
    lharrbig = np.zeros((1, 2, nsample*nsample), np.float_)
    
    print("entering loop")
    for i in xrange(Npix):
        rharrbig[0, 0, :] = measpart0[i] - truepart0
        rharrbig[1, 0, :] = measpart1[i] - truepart1
        lharrbig[0, 0, :] = measpart0[i] - truepart0
        lharrbig[0, 1, :] = measpart1[i] - truepart1
    
        outfast[i, :] = (1/(np.pi*sigpGsq[i]))*np.exp(-0.5*np.einsum('ij...,jk...->ik...', lharrbig, np.einsum('ij...,jk...->ik...', invsig[i, :, :], rharrbig)))
    time1 = time.time()
    print("fast version took ", time1 - time0, "seconds")
    
    if plotrandomsample == True:
        plot_randomsample_posteriors(pmeas, psimeas, p0_all, psi0_all, outfast, cmap = "jet", overplotmeas = True, sigpGsq = sigpGsq)
    
    return outfast, rharrbig, lharrbig, sigpGsq, invsig, i

def project_angles(firstnpoints = 1000):
    """
    Project angles from Equatorial, B-field, IAU Definition -> Galactic, Polarization Angle, Planck Definition
    Note: projected angles are still equally spaced -- no need to re-interpolate
    """
    zero_thetas = fits.getdata("/Volumes/DataDavy/Planck/projected_angles/theta_0.0_Equ_inGal.fits")
    thets = RHT_tools.get_thets(75)
    
    if firstnpoints > 0:
        Npix = firstnpoints
    else:
        Nside=2048
        Npix=12*Nside**2
    
    # This array is a strange fits dtype
    thets_EquinGal = np.mod(np.asarray(zero_thetas[:Npix]).reshape(Npix, 1).astype(np.float_) - thets, np.pi)
    
    return thets_EquinGal
    
def project_angle0_db(wlen = 75, nest=True):
    """
    Project angles from Equatorial, B-field, IAU Definition -> Galactic, Polarization Angle, Planck Definition
    Store only 0-angle in SQL Database by healpix id. Will create other angles on the fly.
    Note: projected angles are still equally spaced -- no need to re-interpolate
    
    if nest : index by hp id in NEST ordering. Otherwise, RING.
    """
    
    # Note that fits.getdata reads this map in incorrectly. 
    zero_thetas = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/projected_angles/theta_0.0_Equ_inGal.fits", nest=False)
    
    # resolution
    Nside = 2048
    Npix = 12*Nside**2
    
    if nest:
        # Convert to NESTED ordering
        zero_thetas = hp.pixelfunc.reorder(zero_thetas, r2n = True)

    # Name table
    tablename = "theta_bin_0_wlen"+str(wlen)

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY, zerotheta FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    if nest:
        conn = sqlite3.connect("theta_bin_0_wlen75_db.sqlite")
    else:
        conn = sqlite3.connect("theta_bin_0_wlen75_db_RING.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()
    
    insertstatement = "INSERT INTO "+tablename+" VALUES (?, ?)"
    
    for _hp_index in xrange(Npix):
        c.execute(insertstatement, [_hp_index, zero_thetas[_hp_index]])    
    
    conn.commit()
    return c
        
def add_hthets(data1, data2):
    """
    Combine two R(theta) arrays
    Overlapping (x, y) points have associated R(theta) arrays summed
    Unique (x, y) points are added to the list
    
    data1 :: this is the array to be *ADDED TO*
    """
    
    # Note: we are in a situation where there will be many more appends than new keys.
    # It is cheaper to try the append and catch the exception than to check for the key existence first
    # (ask forgiveness, not permission)
    
    for key in data2.keys():
        try:
            data1[key] += data2[key]
            
        except KeyError:
            data1[key] = data2[key]
    
    return data1
    
def store_weights_as_dict():    
    # Pull in each projected theta bin
    projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"

    # Output filename
    projected_data_dictionary_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_dictionary.p"

    # resolution
    Nside = 2048
    Npix = 12*Nside**2

    # These are the healpix indices. They will be the keys in our dictionary.
    hp_index = np.arange(Npix)

    nthets = 165 

    total_weights = {}

    for _thetabin_i in xrange(1):
        time0 = time.time()
        projected_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_"+str(_thetabin_i)+".fits"
        projdata = fits.getdata(projected_fn)

        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements".format(len(nonzero_index)))

        # Make arrays of len(nthets) which contain the RHT weights at specified thetabin.
        rht_weights = np.zeros((len(nonzero_index), nthets), np.float_)
        rht_weights[:, _thetabin_i] = projdata[nonzero_index]

        # Add these weights to all the other weights in a dictionary
        indexed_weights = dict(zip(nonzero_index, rht_weights))
        total_weights = add_hthets(total_weights, indexed_weights)
        time1 = time.time()

        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))
    
    # pickle the entire dictionary
    pickle.dump( total_weights, open( projected_data_dictionary_fn, "wb" ) )

    #return total_weights

def plasz_P_to_database(Nside = 2048):
    """
    Puts Colin's implementation of debiased P from Plaszczynski et al into an SQL db
    Indexed using NESTED healpix indices
    """
    Npix = 12*Nside**2
    
    # Place data into array
    usedata = np.zeros((2, Npix), np.float_)
    
    data_root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/"
    usedata[0, :] = hp.fitsfunc.read_map(data_root + "HFI_SkyMap_353_2048_R2.00_full_PdebiasPlasz_RING.fits")
    usedata[1, :] = hp.fitsfunc.read_map(data_root + "HFI_SkyMap_353_2048_R2.00_full_sigPdebiasPlasz_RING.fits")
    
    # Reorder data to NEST ordering
    usedata_nest = hp.pixelfunc.reorder(usedata, r2n = True)
    
    tablename = "P_sigP_Plasz_debias_Nside_2048_Galactic"
    
    value_names = ["Pdebias", "sigPdebias"]
    
    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY, Pdebias FLOAT DEFAULT 0.0, sigPdebias FLOAT DEFAULT 0.0);"
    
    #conn = sqlite3.connect(":memory:")
    conn = sqlite3.connect("P_sigP_Plasz_debias_Nside_2048_Galactic_db.sqlite")
    
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()
    
    insertstatement = "INSERT INTO "+tablename+" VALUES (?, ?, ?)"
    
    print("Beginning database creation")
    for _hp_index in xrange(Npix):
        c.execute(insertstatement, [i for i in itertools.chain([_hp_index], usedata[:, _hp_index])])    
    
    conn.commit()

def QU_RHT_Gal_to_database(sigma=30, smooth=True):
    """
    Project QRHT, URHT data that's been rotated to Galactic coordinates.
    indexed by NEST healpix indices. Covers GALFA allsky.
    """
    Nside=2048
    Npix = 12*Nside**2
     
    if smooth is True:
        tqu_Gal_fn = "../data/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_sig"+str(sigma)+"_Gal.fits"
        tqu_sq_Gal_fn = "../data/TQUsq_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_sig"+str(sigma)+"_Gal.fits"
    else:
        tqu_Gal_fn = "../data/TQU_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Gal.fits"
        tqu_sq_Gal_fn = "../data/TQUsq_RHT_Planck_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Gal.fits"
    TRHTGal_ring, QRHTGal_ring, URHTGal_ring = hp.fitsfunc.read_map(tqu_Gal_fn, field=(0,1,2))
    TRHTsqGal_ring, QRHTsqGal_ring, URHTsqGal_ring = hp.fitsfunc.read_map(tqu_sq_Gal_fn, field=(0,1,2))
    
    # Place data into array
    usedata = np.zeros((4, Npix), np.float_)
    
    TRHTGal = hp.pixelfunc.reorder(TRHTGal_ring, r2n = True)
    usedata[0, :] = hp.pixelfunc.reorder(QRHTGal_ring, r2n = True)
    usedata[1, :] = hp.pixelfunc.reorder(URHTGal_ring, r2n = True)
    usedata[2, :] = hp.pixelfunc.reorder(QRHTsqGal_ring, r2n = True)
    usedata[3, :] = hp.pixelfunc.reorder(URHTsqGal_ring, r2n = True)
    
    Tmask = copy.copy(TRHTGal)
    Tmask[np.where(TRHTGal >= 0.5)] = 1
    Tmask[np.where(TRHTGal < 0.5)] = 0
    
    # The healpix indices we keep will be the ones where there is nonzero data
    nonzero_index = np.nonzero(Tmask)[0]
    print("there are {} nonzero elements".format(len(nonzero_index)))

    if smooth is True:
        tablename = "QURHT_QURHTsq_Gal_pol_ang_chS1004_1043_sig"+str(sigma)
    else:    
        tablename = "QURHT_QURHTsq_Gal_pol_ang_chS1004_1043"
    
    value_names = ["QRHT", "URHT", "QRHTsq", "URHTsq"]
    
    column_names = " FLOAT DEFAULT 0.0,".join(value_names)
    
    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"
    
    # there are 4 values + hp_index to store in this db
    numvalues = 5
    insertstatement = "INSERT INTO "+tablename+" VALUES ("+",".join('?'*numvalues)+")"

    if smooth is True:
        conn = sqlite3.connect("QURHT_QURHTsq_sig"+str(sigma)+"_Gal_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Nside_2048_Galactic_db.sqlite")
    else:
        conn = sqlite3.connect("QURHT_QURHTsq_Gal_pol_ang_GALFA_HI_allsky_coadd_chS1004_1043_w75_s15_t70_Nside_2048_Galactic_db.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()

    print("Beginning database creation")
    for _hp_index in nonzero_index:
        #try:
        c.execute(insertstatement, [i for i in itertools.chain([_hp_index], usedata[:, _hp_index])])    
     
    conn.commit()
    
def planck_data_to_database(Nside = 2048, covdata = True):

    # This is in RING order.
    map353Gal, cov353Gal = get_Planck_data(Nside = Nside)
    Npix = 12*Nside**2
    
    # Want pixel indices in NESTED order:
    #hp_indices = hp.ring2nest(Nside, len(map353Gal[0, :]))
    
    # Convert data to NESTED order:
    if covdata is True:
        usedata = np.asarray(cov353Gal)
        usedata = usedata.reshape(9, Npix)
        usedata = hp.pixelfunc.reorder(usedata, r2n = True)
        usedata = np.asarray(usedata)
    else:
        usedata = hp.pixelfunc.reorder(map353Gal, r2n = True)
        usedata = np.asarray(usedata)
    
    # Should also do this for covariance matrix data...
    
    # map353Gal contains T, Q, U information
    if covdata is True:
        tablename = "Planck_Nside_2048_cov_Galactic"
    else:
        tablename = "Planck_Nside_2048_TQU_Galactic"
    
    # Comma separated list of nthets column names
    if covdata is True:
        value_names = ["TT", "TQ", "TU", "TQa", "QQ", "QU", "TU1", "QUa", "UU"]
    else:
        value_names = ["T", "Q", "U"]
    
    column_names = " FLOAT DEFAULT 0.0,".join(value_names)

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"
    
    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    if covdata is True:
        conn = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
    else:
        conn = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()
    
    if covdata is True:
        numvalues = 10
    else:
        numvalues = 4
    
    insertstatement = "INSERT INTO "+tablename+" VALUES ("+",".join('?'*numvalues)+")"
    
    print("Beginning database creation")
    for _hp_index in xrange(Npix):
        #try:
        c.execute(insertstatement, [i for i in itertools.chain([_hp_index], usedata[:, _hp_index])])    
        #except:
        #    print(_hp_index, map353Gal[:, _hp_index])
    
    conn.commit()

def project_allsky_thetaweights_to_database(update = False):
    """
    Projects allsky weights to healpix Galactic.
    Writes all projected weights from region to an SQL database.
    NOTE :: id = primary key, pixel index in NESTED order
    """
    
    # Pull in each unprojected theta bin
    unprojected_root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/"

    # Output filename
    projected_data_dictionary_fn = unprojected_root + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_dictionary.p"

    # Full GALFA file header for projection
    galfa_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyWide/GALFA_HI_W_S1024_V0000.4kms.fits"
    galfa_hdr = fits.getheader(galfa_fn)

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")
    if "in" in value_names: value_names.remove("in")
    if "if" in value_names: value_names.remove("if")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights_allsky"
    #tablename = "RHT_weights_allsky_testtrans"

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    #conn = sqlite3.connect("/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/allsky_RHTweights_db_testtrans.sqlite")
    conn = sqlite3.connect("/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/allsky_RHTweights_db.sqlite")
    c = conn.cursor()
    
    if update is True:
        print("table already created -- this is simply an update")
    else:
        c.execute(createstatement)
        conn.commit()

    # try setting isolation level
    #conn.isolation_level = None

    for _thetabin_i in range(23, nthets, 1):
    #for _thetabin_i in xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        #unprojected_fn = unprojected_root + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_"+str(_thetabin_i)+".fits"
        #unprojdata = fits.getdata(unprojected_fn)

        # Project data to hp galactic
        #projdata, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(unprojdata, galfa_hdr)
        #print("Data successfully projected")
        
        projected_fn = unprojected_root + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_"+str(_thetabin_i)+"_healpixproj.fits"
        projdata = fits.getdata(projected_fn)
    
        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements in thetabin {}".format(len(nonzero_index), _thetabin_i))
        
        # Try wrapping this in a transaction
        #c.execute("begin")
        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(i, projdata[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(projdata[i], i) for i in nonzero_index])
        #c.execute("commit")
        
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))

    conn.close()  
    
def write_allsky_singlevel_thetaweights_to_database_RADEC(update = False, velstr="S0974_0978"):
    """
    Writes all projected weights from region to an SQL database.
    primary key is flattened index of array with shape of total galfa area
    This version is for a *single* velocity slice
    """          
    
    # Pull in each unprojected theta bin
    unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velstr+"/"

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")
    if "in" in value_names: value_names.remove("in")
    if "if" in value_names: value_names.remove("if")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights_allsky_"+velstr

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    conn = sqlite3.connect(unprojected_root + "GALFA_HI_allsky_"+velstr+"_RADEC_w75_s15_t70_RHTweights_db.sqlite")
    c = conn.cursor()
    
    if update is True:
        print("table already created -- this is simply an update")
    else:
        c.execute(createstatement)
        conn.commit()
        
    # Shape of the all-sky data
    #nyfull = 2432
    #nxfull = 21600
    #fulldata = np.arange(nyfull*nxfull).reshape(nyfull, nxfull)
        
    for _thetabin_i in xrange(nthets):
        time0 = time.time()
        
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "GALFA_HI_W_"+velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)
        
        nonzero_index = np.nonzero(unprojdata)[0]
        print("there are {} nonzero elements in thetabin {}".format(len(nonzero_index), _thetabin_i))
        
        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(i, unprojdata.flat[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(unprojdata.flat[i], i) for i in nonzero_index])
    
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} minutes".format(_thetabin_i, (time1 - time0)/60.))

    conn.close()
    
def write_allsky_singlevel_thetaweights_to_database_RADEC_indx(update = False, velstr="S0974_0978"):
    """
    Writes all projected weights from region to an SQL database.
    primary key is flattened index of array with shape of total galfa area
    id created from allpix = np.arange(2432*21600).reshape(2432, 21600)
    This version is for a *single* velocity slice
    """          
    
    # Pull in each unprojected theta bin
    unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velstr+"/"

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "do" in value_names: value_names.remove("do")
    if "id" in value_names: value_names.remove("id")
    if "is" in value_names: value_names.remove("is")
    if "in" in value_names: value_names.remove("in")
    if "if" in value_names: value_names.remove("if")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights_allsky_"+velstr

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    conn = sqlite3.connect(unprojected_root + "GALFA_HI_allsky_"+velstr+"_RADEC_indx_w75_s15_t70_RHTweights_db.sqlite")
    c = conn.cursor()
    
    if update is True:
        print("table already created -- this is simply an update")
    else:
        c.execute(createstatement)
        conn.commit()
        
    # Shape of the all-sky data
    #nyfull = 2432
    #nxfull = 21600
    allpix = fits.getdata("/disks/jansky/a/users/goldston/susan/Wide_maps/allpix_allsky.fits")
        
    for _thetabin_i in xrange(nthets):
        time0 = time.time()
        
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "GALFA_HI_W_"+velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)
        
        # must flatten data array before finding nonzero elements
        nonzero_index = np.nonzero(unprojdata.flatten())[0]
        print("there are {} nonzero elements in thetabin {}".format(len(nonzero_index), _thetabin_i))
        
        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(allpix.flat[i], unprojdata.flat[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(unprojdata.flat[i], allpix.flat[i]) for i in nonzero_index])
    
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} minutes".format(_thetabin_i, (time1 - time0)/60.))

    conn.close()

def project_allsky_singlevel_thetaweights_to_database(update = False, velstr="S0974_0978"):
    """
    Projects allsky weights to healpix Galactic.
    Writes all projected weights from region to an SQL database.
    NOTE :: id = primary key, pixel index in NESTED order
    This version is for a *single* velocity slice
    """
    
    # Pull in each unprojected theta bin
    unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velstr+"/"

    # Full GALFA file header for projection
    #galfa_hdr = fits.getheader("/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRcorr/GALFA-HI_NHI_VLSR-90+90kms/data/GALFA-HI_NHI_VLSR-90+90kms.fits")
    galfa_hdr = fits.getheader("/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHI_VLSR-90+90kms.fits")

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")
    if "in" in value_names: value_names.remove("in")
    if "if" in value_names: value_names.remove("if")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights_allsky_"+velstr
    #tablename = "RHT_weights_allsky_testtrans"

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    #conn = sqlite3.connect("/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/allsky_RHTweights_db_testtrans.sqlite")
    conn = sqlite3.connect(unprojected_root + "GALFA_HI_allsky_"+velstr+"_w75_s15_t70_RHTweights_db.sqlite")
    c = conn.cursor()
    
    if update is True:
        print("table already created -- this is simply an update")
    else:
        c.execute(createstatement)
        conn.commit()

    # try setting isolation level
    #conn.isolation_level = None

    #for _thetabin_i in range(23, nthets, 1):
    for _thetabin_i in xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "GALFA_HI_W_"+velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)

        # Check if projected data has already been saved
        proj_fn_out = unprojected_root+"/hp_projected/"+"GALFA_HI_W_"+velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(_thetabin_i)+"_healpixproj.fits"
        
        if os.path.isfile(proj_fn_out):
            projdata = hp.fitsfunc.read_map(projected_fn)
            print("Loaded projected data with size {}".format(projdata.shape))
        else:
            # Project data to hp galactic
            projdata, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(unprojdata, galfa_hdr, local=False)
            print("Data successfully projected")
            
            hp.fitsfunc.write_map(proj_fn_out, projdata)
    
        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements in thetabin {}".format(len(nonzero_index), _thetabin_i))
        
        # Try wrapping this in a transaction
        #c.execute("begin")
        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(i, projdata[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(projdata[i], i) for i in nonzero_index])
        #c.execute("commit")
        
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))

    conn.close()
    
def project_allsky_vel_weighted_int_thetaweights_to_database(update = False):
    """
    Projects allsky weighted integrated thetaweights to healpix Galactic.
    Writes all projected weights from region to an SQL database.
    NOTE :: id = primary key, pixel index in NESTED order
    This version is for single_theta_S0974_1073_sum
    """
    
    # Pull in each unprojected theta bin
    unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/weighted_single_theta_maps/single_theta_S0974_1073_sum/"

    # Full GALFA file header for projection
    galfa_hdr = fits.getheader("/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHI_VLSR-90+90kms.fits")

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")
    if "in" in value_names: value_names.remove("in")
    if "if" in value_names: value_names.remove("if")
    if "do" in value_names: value_names.remove("do")
    if "id" in value_names: value_names.remove("id")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights_allsky"

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    conn = sqlite3.connect(unprojected_root + "GALFA_HI_allsky_weighted_int_S0974_1073_w75_s15_t70_RHTweights_db_fast.sqlite")
    c = conn.cursor()
    
    if update is True:
        print("table already created -- this is simply an update")
    else:
        c.execute(createstatement)
        conn.commit()

    #for _thetabin_i in range(23, nthets, 1):
    for _thetabin_i in xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        #unprojected_fn = unprojected_root + "weighted_rht_power_0974_1073_thetabin_"+str(_thetabin_i)+".fits"
        #unprojdata = fits.getdata(unprojected_fn)

        # Project data to hp galactic
        #projdata, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(unprojdata, galfa_hdr, local=False)
        #print("Data successfully projected")
        
        # load in already projected data
        projected_fn = unprojected_root + "weighted_rht_power_0974_1073_thetabin_"+str(_thetabin_i)+"_healpixproj_nanmask.fits"
        projdata = fits.getdata(projected_fn)
    
        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0
        projdata[projdata < 0] = 0
        projdata[np.isnan(projdata)] = 0
        projdata[np.where(projdata == None)] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements in thetabin {}".format(len(nonzero_index), _thetabin_i))
        
        # Try wrapping this in a transaction
        #c.execute("begin")
        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(i, projdata[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(projdata[i], i) for i in nonzero_index])
        #c.execute("commit")
        
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))

    conn.close()
    
def intRHT_QU_maps_per_vel(velstr="S0974_0978"):
    
    # Pull in each unprojected theta bin
    unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velstr+"/"

    # Shape of the all-sky data
    nyfull = 2432
    nxfull = 21600
    intRHT = np.zeros((nyfull, nxfull), np.float_)
    QRHT = np.zeros((nyfull, nxfull), np.float_)
    URHT = np.zeros((nyfull, nxfull), np.float_)
    QRHTsq = np.zeros((nyfull, nxfull), np.float_)
    URHTsq = np.zeros((nyfull, nxfull), np.float_)
    
    thets = RHT_tools.get_thets(75)
    
    thetshist = np.zeros(len(thets))

    for _thetabin_i in xrange(165):
        time0 = time.time()
    
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "GALFA_HI_W_"+velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)  
        
        intRHT += unprojdata
        QRHT += np.cos(2*thets[_thetabin_i])*unprojdata
        URHT += np.sin(2*thets[_thetabin_i])*unprojdata
        QRHTsq += np.cos(2*thets[_thetabin_i])**2*unprojdata
        URHTsq += np.sin(2*thets[_thetabin_i])**2*unprojdata
        
        thetshist[_thetabin_i] = np.nansum(unprojdata)
        
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))
    
    hdr = fits.getheader(unprojected_fn)    
    fits.writeto(unprojected_root+"intrht_"+velstr+".fits", intRHT, hdr)
    fits.writeto(unprojected_root+"QRHT_"+velstr+".fits", QRHT, hdr)
    fits.writeto(unprojected_root+"URHT_"+velstr+".fits", URHT, hdr)
    fits.writeto(unprojected_root+"QRHTsq_"+velstr+".fits", QRHTsq, hdr)
    fits.writeto(unprojected_root+"URHTsq_"+velstr+".fits", URHTsq, hdr)
    np.save(unprojected_root+"thets_hist_"+velstr+".fits", thetshist)

def make_single_theta_int_vel_map(thetabin=0):
    """
    Make a single theta map that has the total weight at that thetabin for all velocities. 
    """
    
    velstrs=["S0974_0978", "S0979_0983", "S0984_0988", "S0989_0993", "S0994_0998", "S0999_1003",
             "S1004_1008", "S1009_1013", "S1014_1018", "S1019_1023", "S1024_1028", "S1029_1033",
             "S1034_1038", "S1039_1043", "S1044_1048", "S1049_1053", "S1054_1058", "S1059_1063",
             "S1064_1068", "S1069_1073", "S1074_1078"]
    
    in_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"
    out_root = in_root + "single_theta_0974_1078_sum/" 
    # Shape of the all-sky data
    nyfull = 2432
    nxfull = 21600
    single_vel_map = np.zeros((nyfull, nxfull), np.float_)   
    
    for _velstr in velstrs:
        single_theta_fn = in_root+_velstr+"/GALFA_HI_W_"+_velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(thetabin)+".fits"
        single_vel_map += fits.getdata(single_theta_fn)    
        print(_velstr, np.nansum(single_vel_map))

    hdr = fits.getheader(single_theta_fn)
    fits.writeto(out_root+"total_rht_power_0974_1078_thetabin_"+str(thetabin)+".fits", single_vel_map)
    
    
def get_RHT_Sstr(starting_vel):

    start_velstr = galfa_name_lookup.get_velstr(starting_vel)
    end_velstr = galfa_name_lookup.get_velstr(starting_vel + 4)
    
    Sstr = "S"+start_velstr+"_"+end_velstr
    
    return Sstr

def make_vel_int_galfa_channel_maps():
    """
    for each of our favorite 5-wide-channel velocity slices, make integrated map
    (channel1 + channel2 + ...)*dv
    """
    
    begin_vel = 974
    end_vel = 1073
    
    nyfull = 2432
    nxfull = 21600 
    
    #cdelt3 in original Wide cube
    cdelt3 = 0.736122839600
    
    galfa_root = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/Allsky_ChanMaps/Wide/"
    out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/channel_maps_for_RHT/newmaps_010318/"
    
    rht_starting_vels = [begin_vel + 5*i for i in xrange((end_vel - begin_vel)//5 + 1)]
    
    for i, _vel in enumerate(rht_starting_vels):
    
        Sstr = get_RHT_Sstr(_vel)
        
        # start with new sumchans for each velocity group
        sumchans = np.zeros((nyfull, nxfull), np.float_)   
    
        for v in np.arange(rht_starting_vels[i], rht_starting_vels[i]+5):
            print(v)
        
            galfa_fn = galfa_root + galfa_name_lookup.get_galfa_W_name(v)
            sumchans += fits.getdata(galfa_fn)
            hdr = fits.getheader(galfa_fn)
        
        fits.writeto(out_root+"channel_map_"+Sstr+".fits", sumchans*cdelt3, hdr)
        
    

def make_weighted_single_theta_int_vel_map(thetabin=0):
    """
    Make a single theta map that has the total weight at that thetabin for all velocities. 
    Weight each velocity by the local HI intensity at that velocity channel.
    
    i.e. map(theta_i) = sum_v (I(v)*R(theta_i))
    """
    
    velstrs=["S0974_0978", "S0979_0983", "S0984_0988", "S0989_0993", "S0994_0998", "S0999_1003",
             "S1004_1008", "S1009_1013", "S1014_1018", "S1019_1023", "S1024_1028", "S1029_1033",
             "S1034_1038", "S1039_1043", "S1044_1048", "S1049_1053", "S1054_1058", "S1059_1063",
             "S1064_1068", "S1069_1073"]#, "S1074_1078"]
    
    in_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"
    in_channel_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/channel_maps_for_RHT/"
    #out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/weighted_single_theta_maps/single_theta_S0974_1073_sum/" 
    out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/weighted_single_theta_maps/single_theta_S0974_1073_sum_corrected_chanmaps/" 
    # Shape of the all-sky data
    nyfull = 2432
    nxfull = 21600
    single_vel_map = np.zeros((nyfull, nxfull), np.float_)   
    
    begin_vel = 974
    end_vel = 1073

    rht_starting_vels = [begin_vel + 5*i for i in xrange((end_vel - begin_vel)//5 + 1)]
    
    for _velstr in velstrs:
        # get RHT power at single theta for this channel
        single_theta_fn = in_root+_velstr+"/GALFA_HI_W_"+_velstr+"_newhdr_SRcorr_w75_s15_t70_theta_"+str(thetabin)+".fits"
        single_theta_map = fits.getdata(single_theta_fn)
        
        # get channel map
        velocity_channel_fn = in_channel_root+"channel_map_"+_velstr+".fits"
        velocity_channel_map = fits.getdata(velocity_channel_fn)
        
        # multiply by channel intensity
        single_vel_map += single_theta_map*velocity_channel_map
        
        print(_velstr, "num pix:", np.nansum(single_vel_map))

    hdr = fits.getheader(velocity_channel_fn)
    fits.writeto(out_root+"weighted_rht_power_0974_1073_thetabin_"+str(thetabin)+".fits", single_vel_map)


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
  
def coadd_QU_maps():
    wlen = 75
    cstep = 5 
    
    # Shape of the all-sky data
    nyfull = 2432
    nxfull = 21600
    Qdata = np.zeros((nyfull, nxfull), np.float_)
    Udata = np.zeros((nyfull, nxfull), np.float_)
    intdata = np.zeros((nyfull, nxfull), np.float_)

    for velnum in np.arange(-10, 10):
    
        # Everything is in chunks of 5 channels. e.g. 1024_1028 includes [1024, 1028] inclusive.
        cstart = 1024 + velnum*cstep
        cstop = cstart + cstep - 1
        s_string, extra_0 = get_extra0_sstring(cstart, cstop)
    
        velrangestring = s_string+str(cstart)+"_"+extra_0+str(cstop)

        in_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/"+velrangestring+"/"
        
        Qdata += fits.getdata(in_root + "QRHT_"+velrangestring+".fits")
        Udata += fits.getdata(in_root + "URHT_"+velrangestring+".fits")
        intdata += fits.getdata(in_root + "intrht_"+velrangestring+".fits")
    
    outhdr = fits.getheader(in_root + "QRHT_"+velrangestring+".fits")
        
    cbegin = 1024 + -10*cstep
    cend = 1024 + 9*cstep
        
    fits.writeto("/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/intrht_coadd_"+str(cbegin)+"_"+str(cend)+".fits", intdata, outhdr)
    fits.writeto("/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/QRHT_coadd_"+str(cbegin)+"_"+str(cend)+".fits", Qdata, outhdr)
    fits.writeto("/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/URHT_coadd_"+str(cbegin)+"_"+str(cend)+".fits", Udata, outhdr)
        
    
def reproject_allsky_data(local=True):
    
    # Pull in each unprojected theta bin
    if local:
        unprojected_root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/"
    else:
        unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/single_theta_maps/single_theta_0974_1078_sum/"
        
    nthets = 165
    
    galfa_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyWide/GALFA_HI_W_S1024_V0000.4kms.fits"
    galfa_hdr = fits.getheader(galfa_fn)
    
    for _thetabin_i in xrange(1):#xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)

        # Project data to hp galactic
        projdata, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(unprojdata, galfa_hdr, nonedata=None, local=local)
        print("Data successfully projected")
        
        projected_fn = unprojected_root + "GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_"+str(_thetabin_i)+"_healpixproj_nanmask.fits"
        
        out_hdr["THETAI"] = _thetabin_i
        out_hdr["VSTART"] = -10
        out_hdr["VSTOP"] = 10
    
        fits.writeto(projected_fn, projdata, out_hdr)
        
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))
        
def reproject_allsky_weighted_data(local=True):
    
    # Pull in each unprojected theta bin
    if local:
        unprojected_root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/"
    else:
        unprojected_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/weighted_single_theta_maps/single_theta_S0974_1073_sum/"
        
    nthets = 165
    
    if local:
        galfa_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyWide/GALFA_HI_W_S1024_V0000.4kms.fits"
    else:
        galfa_fn = "/disks/jansky/a/users/goldston/zheng/151019_NHImaps_SRcorr/data/GNHImaps_SRCORR_final/NHImaps/GALFA-HI_NHI_VLSR-90+90kms.fits"

        
    galfa_hdr = fits.getheader(galfa_fn)
    
    for _thetabin_i in np.arange(150, 166):
        time0 = time.time()
    
        # Load in single-theta backprojection
        unprojected_fn = unprojected_root + "weighted_rht_power_0974_1073_thetabin_"+str(_thetabin_i)+".fits"
        unprojdata = fits.getdata(unprojected_fn)

        # Project data to hp galactic
        projdata, out_hdr = rht_to_planck.interpolate_data_to_hp_galactic(unprojdata, galfa_hdr, local=local, nonedata=None)
        print("Data successfully projected")
        
        
        projected_fn = unprojected_root + "weighted_rht_power_0974_1073_thetabin_"+str(_thetabin_i)+"_healpixproj_nanmask.fits"
        
        out_hdr["THETAI"] = _thetabin_i
        out_hdr["VSTART"] = 974
        out_hdr["VSTOP"] = 1073
    
        fits.writeto(projected_fn, projdata, out_hdr)
        
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))
    
def test_faster_db_creation():
    # Pull in each projected theta bin
    projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"

    # Output filename
    projected_data_dictionary_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_dictionary.p"

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # name table
    tablename = "test_RHT_weights"
    
    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    conn = sqlite3.connect("allweights_db.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()

    for _thetabin_i in xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        projected_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_"+str(_thetabin_i)+".fits"
        projdata = fits.getdata(projected_fn)

        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements".format(len(nonzero_index)))
        
        

def projected_thetaweights_to_database():
    """
    Writes all projected weights from region to an SQL database.
    Code to store weights as pickled dict can't be run on laptop but this *can*
    For SC_241, SQL database is 1.4Gb on disk rather than 7.8Gb for pickled dict
    NOTE :: id = primary key, pixel index in NESTED order
    """
    
    # Pull in each projected theta bin
    projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"

    # Output filename
    projected_data_dictionary_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_dictionary.p"

    nthets = 165 

    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "RHT_weights"

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    #conn = sqlite3.connect(":memory:")
    conn = sqlite3.connect("allweights_db.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()

    for _thetabin_i in xrange(nthets):
        time0 = time.time()
    
        # Load in single-theta backprojection
        projected_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_"+str(_thetabin_i)+".fits"
        projdata = fits.getdata(projected_fn)

        # Some data stored as -999 for 'none'
        projdata[projdata == -999] = 0

        # The healpix indices we keep will be the ones where there is nonzero data
        nonzero_index = np.nonzero(projdata)[0]
        print("there are {} nonzero elements".format(len(nonzero_index)))

        # Either inserts new ID with given value or ignores if id already exists 
        c.executemany("INSERT OR IGNORE INTO "+tablename+" (id, "+value_names[_thetabin_i]+") VALUES (?, ?)", [(i, projdata[i]) for i in nonzero_index])
    
        # Inserts data to new ids
        c.executemany("UPDATE "+tablename+" SET "+value_names[_thetabin_i]+"=? WHERE id=?", [(projdata[i], i) for i in nonzero_index])
    
        conn.commit()
    
        time1 = time.time()
        print("theta bin {} took {} seconds".format(_thetabin_i, time1 - time0))

    conn.close()
    
def get_largest_rht_id(rht_cursor):
    max_id = rht_cursor.execute("SELECT id from RHT_weights ORDER BY ab DESC LIMIT 1").fetchone()
    
    return max_id
    
def get_all_rht_ids(rht_cursor):
    all_ids = rht_cursor.execute("SELECT id from RHT_weights").fetchall()
    
    return all_ids
    
def get_all_P_sigP_debias(plasz_P_cursor):
    all_Pdebias = plasz_P_cursor.execute("SELECT Pdebias FROM P_sigP_Plasz_debias_Nside_2048_Galactic").fetchall()
    all_Pdebiassig = plasz_P_cursor.execute("SELECT sigPdebias FROM P_sigP_Plasz_debias_Nside_2048_Galactic").fetchall()
    
    return all_Pdebias, all_Pdebiassig
    
def SC_241_posteriors(map353Gal = None, cov353Gal = None, firstnpoints = 1000):
    """
    Calculate 2D Bayesian posteriors for Planck data.
    This is specifically for SC_241 (region from previous paper) for now.
    Uses projected and re-pixelized RHT data: Planck (non-IAU) polarization angle definition,
    Galactic coordinates, Healpix.
    """
    # resolution
    Nside = 2048
    Npix = 12*Nside**2
    #if map353Gal is None:
    #    map353Gal, cov353Gal = get_Planck_data(Nside = Nside)
    
    # Create grid of psi0's and p0's to sample
    nsample = 165
    psi0_all = np.linspace(0, np.pi, nsample)
    p0_all = np.linspace(0, 1.0, nsample)
        
    # Get Planck data from database
    planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
    planck_tqu_cursor = planck_tqu_db.cursor()
    
    planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
    planck_cov_cursor = planck_cov_db.cursor()

    # likelihood = planck-only posterior
    #likelihood = Planck_posteriors(map353Gal = map353Gal, cov353Gal = cov353Gal, firstnpoints = firstnpoints)

    # Planck-projected RHT data for prior stored as SQL db
    rht_db = sqlite3.connect("allweights_db.sqlite")
    rht_cursor = rht_db.cursor()
    tablename = "RHT_weights"
    
    # Projected angle bins
    #theta_bins_gal = project_angles(firstnpoints = firstnpoints)
    
    return likelihood
    
def latex_formatter(x, pos):
    return "${0:.1f}$".format(x)

def plot_bayesian_components(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all, npsample = 165, npsisample = 165):
    fig = plt.figure(figsize = (14, 4), facecolor = "white")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    pp = Posterior(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all)
    
    cmap = "cubehelix"
    im1 = ax1.imshow(pp.planck_likelihood, cmap = cmap)
    ax1.set_title(r"$\mathrm{Planck}$ $\mathrm{Likelihood}$", size = 20)
    div = make_axes_locatable(ax1)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax, format=ticker.FuncFormatter(latex_formatter))

    im2 = ax2.imshow(pp.normed_prior, cmap = cmap)
    ax2.set_title(r"$\mathrm{RHT}$ $\mathrm{Prior}$", size = 20)
    div = make_axes_locatable(ax2)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax, format=ticker.FuncFormatter(latex_formatter))

    im3 = ax3.imshow(pp.normed_posterior, cmap = cmap)
    ax3.set_title(r"$\mathrm{Posterior}$", size = 20)
    div = make_axes_locatable(ax3)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im3, cax=cax, format=ticker.FuncFormatter(latex_formatter))
    
    axs = [ax1, ax2, ax3]
    for ax in axs:
        ax.set_xlabel(r"$\mathrm{p_0}$", size = 20)
        ax.set_ylabel(r"$\psi_0$", size = 20)
        ax.set_xticks(np.arange(len(p0_all))[::30])
        ax.set_xticklabels([r"${0:.2f}$".format(p0) for p0 in np.round(p0_all[::20], decimals = 2)])
        ax.set_yticks(np.arange(len(psi0_all))[::20])
        ax.set_yticklabels([r"${0:.1f}$".format(psi0) for psi0 in np.round(np.degrees(psi0_all[::20]), decimals = 2)])
    
    plt.subplots_adjust(wspace = 0.8)
    
def plot_bayesian_components_from_posterior(pp):
    fig = plt.figure(figsize = (14, 4), facecolor = "white")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    psi0_all = pp.sample_psi0
    p0_all = pp.sample_p0
    
    cmap = "cubehelix"
    im1 = ax1.imshow(pp.planck_likelihood, cmap = cmap)
    ax1.set_title(r"$\mathrm{Planck}$ $\mathrm{Likelihood}$", size = 20)
    div = make_axes_locatable(ax1)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax, format=ticker.FuncFormatter(latex_formatter))

    im2 = ax2.imshow(pp.normed_prior, cmap = cmap)
    ax2.set_title(r"$\mathrm{RHT}$ $\mathrm{Prior}$", size = 20)
    div = make_axes_locatable(ax2)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax, format=ticker.FuncFormatter(latex_formatter))

    im3 = ax3.pcolor(p0_all, psi0_all, pp.normed_posterior, cmap = cmap)

    #im3 = ax3.imshow(pp.normed_posterior, cmap = cmap)
    ax3.set_title(r"$\mathrm{Posterior}$", size = 20)
    div = make_axes_locatable(ax3)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im3, cax=cax, format=ticker.FuncFormatter(latex_formatter))
    
    
    #maxval = np.nanmax(B2D)
    #levels = np.asarray([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9])*maxval
    #obj = ax.contour(p0s, psi0s, B2D, levels, extend = "both", colors = "gray")
    #ax3.set_xlim(p0_all[0], p0_all[-1])
    #ax3.set_ylim(psi0_all[0], psi0_all[-1])
    
    axs = [ax1, ax2]#, ax3]
    for ax in axs:
        ax.set_xlabel(r"$\mathrm{p_0}$", size = 20)
        ax.set_ylabel(r"$\psi_0$", size = 20)
        ax.set_xticks(np.arange(len(p0_all))[::30])
        ax.set_xticklabels([r"${0:.2f}$".format(p0) for p0 in np.round(p0_all[::20], decimals = 2)])
        ax.set_yticks(np.arange(len(psi0_all))[::20])
        ax.set_yticklabels([r"${0:.1f}$".format(psi0) for psi0 in np.round(np.degrees(psi0_all[::20]), decimals = 2)])
    
    plt.subplots_adjust(wspace = 0.8)
    
    return ax1, ax2, ax3

def plot_bayesian_posterior_from_posterior(pp, ax, cmap = "cubehelix"):
    psi0_all = pp.sample_psi0
    p0_all = pp.sample_p0
    
    #im1 = ax.imshow(pp.planck_likelihood, cmap = cmap)
    im1 = ax.pcolor(p0_all, psi0_all, pp.normed_posterior, cmap = cmap)
    ax.set_title(r"$\mathrm{Planck}$ $\mathrm{Likelihood}$", size = 20)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="15%", pad=0.05)
    cbar = plt.colorbar(im1, cax=cax, format=ticker.FuncFormatter(latex_formatter))

    ax.set_xlabel(r"$\mathrm{p_0}$", size = 20)
    ax.set_ylabel(r"$\psi_0$", size = 20)
    #ax.set_xticks(np.arange(len(p0_all))[::30])
    #ax.set_xticklabels([r"${0:.2f}$".format(p0) for p0 in np.round(p0_all[::20], decimals = 2)])
    #ax.set_yticks(np.arange(len(psi0_all))[::20])
    #ax.set_yticklabels([r"${0:.1f}$".format(psi0) for psi0 in np.round(np.degrees(psi0_all[::20]), decimals = 2)])
    
    
def single_posterior(hp_index, wlen = 75):
    
    # Planck covariance database
    planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
    planck_cov_cursor = planck_cov_db.cursor()
    
    # Planck TQU database
    planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
    planck_tqu_cursor = planck_tqu_db.cursor()
    
    # Planck-projected RHT database
    rht_db = sqlite3.connect("allweights_db.sqlite")
    rht_cursor = rht_db.cursor()

    # Create psi0 sampling grid
    psi0_sample_db = sqlite3.connect("theta_bin_0_wlen"+str(wlen)+"_db.sqlite")
    psi0_sample_cursor = psi0_sample_db.cursor()    
    zero_theta = psi0_sample_cursor.execute("SELECT zerotheta FROM theta_bin_0_wlen75 WHERE id = ?", (hp_index,)).fetchone()

    # Create array of projected thetas from theta = 0
    thets = RHT_tools.get_thets(wlen)
    psi0_all = np.mod(zero_theta - thets, np.pi)
    
    # Grab debiased P, sigma_P from Colin's implementation of Plaszczynski et al
    plasz_P_db = sqlite3.connect("P_sigP_Plasz_debias_Nside_2048_Galactic_db.sqlite")
    plasz_P_cursor = plasz_P_db.cursor()
    (Pdebias, Pdebiassig) = plasz_P_cursor.execute("SELECT Pdebias, sigPdebias FROM P_sigP_Plasz_debias_Nside_2048_Galactic WHERE id = ?", (hp_index,)).fetchone()
    
    # We will sample P on a grid from -7 sigma to +7 sigma. Current implementation assumes sigma_I = 0
    numsig = 7
    beginP = Pdebias - numsig*Pdebiassig
    endP = Pdebias + numsig*Pdebiassig
    print(beginP, endP, Pdebias, Pdebiassig)
    sample_P = np.linspace(beginP, endP, len(psi0_all))
    
    # Turn sample of P into sample of p by dividing by I (p = P/I)
    I0 = planck_tqu_cursor.execute("SELECT T FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    p0_all = sample_P/I0
    
    print(I0)
    
    # Also get naive P
    Qmeas = planck_tqu_cursor.execute("SELECT Q FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    Umeas = planck_tqu_cursor.execute("SELECT U FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    Pnaive = np.sqrt(Qmeas[0]**2 + Umeas[0]**2)
    
    #print("Naive P is {}".format(Pnaive))
    print("Naive p is {}".format(Pnaive/I0))
    #print("Debiased P is {}".format(Pdebias))
    #print("Debiased p is {}".format(Pdebias/I0[0]))
    print("Naive psi is {}".format(np.mod(0.5*np.arctan2(Umeas, Qmeas), np.pi)))
    
    minPnaive = Pnaive - numsig*Pdebiassig
    maxPnaive = Pnaive + numsig*Pdebiassig
    
    beginP = max((0.0, minPnaive))
    endP = min((1.0, maxPnaive))
    print(beginP, endP, maxPnaive, minPnaive, Pdebiassig)
    
    sample_P = np.linspace(beginP, endP, len(psi0_all))
    p0_all_naive = sample_P/I0
    
    # Sampling should cover naive as well as debiased P in range

    #posterior = Posterior(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all)
    posterior_naive = Posterior(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all_naive, psi0_all)
    #plot_bayesian_components(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all, npsample = 165, npsisample = 165)
    plot_bayesian_components(hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all_naive, psi0_all, npsample = 165, npsisample = 165)
    
    return posterior_naive, p0_all_naive, psi0_all

def center_psi_measurement(array, sample_psi, psi_meas):

    # Find index of value closest to psi_meas - pi/2
    psi_meas_indx = np.abs(sample_psi - np.mod(psi_meas - np.pi/2, np.pi)).argmin()
    
    print(psi_meas_indx, sample_psi[psi_meas_indx])
    
    rolled_sample_psi = np.roll(sample_psi, -psi_meas_indx)
    rolled_array = np.roll(array, -psi_meas_indx, axis = 0)
    
    return rolled_array, rolled_sample_psi
    
def roll_zero_to_pi(array, sample_psi):

    # Find index of value closest to 0
    psi_0_indx = np.abs(sample_psi).argmin()
    
    print(psi_0_indx, sample_psi[psi_0_indx])
    
    rolled_sample_psi = np.roll(sample_psi, -psi_0_indx)
    rolled_array = np.roll(array, -psi_0_indx, axis = 0)
    
    return rolled_array, rolled_sample_psi

def roll_RHT_zero_to_pi(rht_data, sample_psi):

    # Find index of value closest to 0
    psi_0_indx = np.abs(sample_psi).argmin()
    
    print("rolling data by", psi_0_indx, sample_psi[psi_0_indx])
    
    rolled_sample_psi = np.roll(sample_psi, -psi_0_indx)
    rolled_array = np.roll(rht_data, -psi_0_indx)
    
    return rolled_array, rolled_sample_psi

def wrap_to_pi_over_2(angles):
    while np.nanmax(angles) > np.pi/2:
        angles[np.where(angles > np.pi/2)] = angles[np.where(angles > np.pi/2)] - np.pi
    while np.nanmin(angles) < -np.pi/2:
        angles[np.where(angles < -np.pi/2)] = angles[np.where(angles < -np.pi/2)] + np.pi
    
    return angles
    
def make_gaussian(len, fwhm = 3, center = None):

    x = np.arange(0, len, 1, float)
    y = x[:, np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    
def test_estimator_gaussians():

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    axs = [ax1, ax2, ax3]
    fwhms = [3, 8, 15]
    for i, fwhm in enumerate(fwhms):
        tp, testpMB, testpsiMB, testpMB1, testpsiMB1, testsample_psi0, testrolled_grid_sample_psi0, testrolled_posterior, p0moment1, psi0moment1 = test_estimator(fakeit = True, fwhm = fwhm)
        
        tp.normed_posterior, tp.sample_psi0 = roll_zero_to_pi(tp.normed_posterior, tp.sample_psi0)
        
        plot_bayesian_posterior_from_posterior(tp, axs[i])
        
        """
        im = axs[i].imshow(testrolled_posterior, cmap = "cubehelix")
        axs[i].plot(testpMB, np.degrees(testpsiMB), '+', ms = 20, color = "white")
        
        axs[i].set_title(r"$\mathrm{Posterior}$", size = 20)
        div = make_axes_locatable(axs[i])
        cax = div.append_axes("right", size="15%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(latex_formatter))
            
        axs[i].set_xlabel(r"$\mathrm{p_0}$", size = 20)
        axs[i].set_ylabel(r"$\psi_0$", size = 20)
        axs[i].set_xticks(np.arange(len(tp.sample_p0))[::30])
        axs[i].set_xticklabels([r"${0:.2f}$".format(p0) for p0 in np.round(tp.sample_p0[::20], decimals = 2)])
        axs[i].set_yticks(np.arange(len(tp.sample_psi0))[::20])
        axs[i].set_yticklabels([r"${0:.1f}$".format(psi0) for psi0 in np.round(np.degrees(tp.sample_psi0[::20]), decimals = 2)])
        """
        
    plt.subplots_adjust(wspace = 0.8)
def test_estimator(fakeit = True, fwhm = 3):

    tp, p0_all, psi0_all = single_posterior(24066112)#3643649)#3173221)
    
    print("TEST BEGINS")
    
    if fakeit is True:
        #test_posterior = np.zeros((165, 165), np.float_) + 0.00001
        #test_posterior[5, 100] = 1000.
        
        test_posterior = make_gaussian(165, fwhm = fwhm, center = [100, 5])
    
        test_sample_psi = np.linspace(0, np.pi, 165)
        test_sample_p = np.linspace(0, 1, 165)
        
        print("test naive p is {}".format(test_sample_p[100]))
        print("test naive psi is {}".format(test_sample_psi[5]))
        
        test_sample_psi = test_sample_psi[::-1]
        test_posterior = test_posterior[::-1, :]
        
        tp.sample_p0 = test_sample_p
        tp.sample_psi0 = test_sample_psi
        
        tp.psi_dx = test_sample_psi[1] - test_sample_psi[0]
        tp.p_dx = test_sample_p[1] - test_sample_p[0]
        
        if tp.psi_dx < 0:
            tp.psi_dx *= -1
            
        tp.posterior = test_posterior
        norm_factor = tp.integrate_highest_dimension(test_posterior, dx = tp.psi_dx)
        norm_factor = tp.integrate_highest_dimension(norm_factor, dx = tp.p_dx)
        print("norm factor is {}".format(norm_factor))
        tp.normed_posterior = tp.posterior/norm_factor
        
    else:
        test_sample_psi = psi0_all
        test_sample_p = p0_all
    
    testpMB, testpsiMB, testpMB1, testpsiMB1, testsample_psi0, testrolled_grid_sample_psi0, testrolled_posterior, p0moment1, psi0moment1 = mean_bayesian_posterior(tp, sample_psi0 = test_sample_psi, sample_p0 = test_sample_p)
    
    print("pMB is {}".format(testpMB))
    print("psiMB is {}".format(testpsiMB))
    
    return tp, testpMB, testpsiMB, testpMB1, testpsiMB1, testsample_psi0, testrolled_grid_sample_psi0, testrolled_posterior, p0moment1, psi0moment1

def test_normalization(posterior_obj, pdx, psidx):
    norm_posterior_test = posterior_obj.integrate_highest_dimension(posterior_obj.normed_posterior, dx = psidx)
    norm_posterior_test = posterior_obj.integrate_highest_dimension(norm_posterior_test, dx = pdx)
    
    print("Normalized posterior is {}".format(norm_posterior_test))
    
    return norm_posterior_test

def mean_bayesian_posterior(posterior_obj, sample_p0 = None, sample_psi0 = None):
    """
    Integrated first order moments of the posterior PDF
    """
    
    posterior = posterior_obj.normed_posterior
    
    sample_p0 = posterior_obj.sample_p0
    sample_psi0 = posterior_obj.sample_psi0
    
    # Wrap to [-pi/2, pi/2] domain
    #sample_psi0 = wrap_to_pi_over_2(sample_psi0)
    
    grid_sample_p0 = np.tile(sample_p0, (len(sample_p0), 1))
    grid_sample_psi0 = np.tile(np.reshape(sample_psi0, (len(sample_psi0), 1)), (1, len(sample_psi0)))
    
    # First moment of p0 map
    p0moment1 = grid_sample_p0*posterior
    
    # Sampling width for p
    pdx = posterior_obj.p_dx
    #pdx = sample_p0[1] - sample_p0[0]
    
    # Reverse psi's so that they ascend
    #sample_psi0 = sample_psi0[::-1]
    #psidx = sample_psi0[1] - sample_psi0[0]
    psidx = posterior_obj.psi_dx
    
    # Test that normed posterior is normed
    norm_posterior_test = test_normalization(posterior_obj, pdx, psidx)
    if norm_posterior_test < 1.0:
        print("")
    
    #if psidx < 0:
    #    psidx *= -1
    
    print("Sampling pdx is {}, psidx is {}".format(pdx, psidx))
    
    # Reverse posterior in the psi dimension as well
    #posterior = posterior[::-1, :]
    
    # Integrate over p
    pMB1 = np.trapz(p0moment1, dx = pdx, axis = 0)
    
    # Integrate over psi
    pMB = np.trapz(pMB1, dx = psidx)
    
    center_psi = False
    if center_psi is True:
        rolled_posterior, rolled_sample_psi = center_psi_measurement(posterior, sample_psi0, posterior_obj.naive_psi)
    else:
        rolled_posterior = posterior
        rolled_sample_psi = sample_psi0
    
    rolled_grid_sample_psi0 = np.tile(np.reshape(rolled_sample_psi, (len(rolled_sample_psi), 1)), (1, len(rolled_sample_psi)))
    
    # First moment of psi0 map
    psi0moment1 = rolled_grid_sample_psi0*rolled_posterior
    
    # Integrate over p
    psiMB1 = np.trapz(psi0moment1, dx = pdx, axis = 0)
    
    # Find index of value closest to pi/2
    #piover2_indx = np.abs(sample_psi0 - np.pi/2).argmin()
    #psiMB1 = np.roll(psiMB1, -piover2_indx, axis = 0)
    #sample_psi0 = np.roll(sample_psi0, -piover2_indx)
    
    # Integrate over psi
    psiMB = np.trapz(psiMB1, dx = psidx)
    
    print("pMB is {}".format(pMB))
    print("psiMB is {}".format(psiMB))
    
    return pMB, psiMB, pMB1, psiMB1, sample_psi0, rolled_grid_sample_psi0, rolled_posterior, p0moment1, psi0moment1
    
def plot_sampled_posterior(hp_index):
    posterior, posterior_naive, p0_all_naive, psi0_all = single_posterior(hp_index)
    pMB, psiMB = mean_bayesian_posterior(posterior_naive.normed_posterior, sample_psi0 = psi0_all, sample_p0 = p0_all_naive)
    
    ax1, ax2, ax3 = plot_bayesian_components_from_posterior(posterior_naive)
    ax3.plot(pMB, np.degrees(psiMB), '+', ms = 20, color = "white")
    
class BayesianComponent():
    """
    Base class for building Bayesian pieces
    Instantiated by healpix index
    """
    
    def __init__(self, hp_index):
        self.hp_index = hp_index
    
    def integrate_highest_dimension(self, field, dx = 1):
        """
        Integrates over highest-dimension axis.
        """
        axis_num = field.ndim - 1
        integrated_field = np.trapz(field, dx = dx, axis = axis_num)
        
        return integrated_field
    

class Prior(BayesianComponent):
    """
    Class for building RHT priors
    """
    
    def __init__(self, hp_index, c, p0_all, psi0_all, reverse_RHT = False):
    
        BayesianComponent.__init__(self, hp_index)
        self.rht_data = c.execute("SELECT * FROM RHT_weights WHERE id = ?", (self.hp_index,)).fetchone()
        
        # Discard first element because it is the healpix id
        self.rht_data = self.rht_data[1:]
        
        self.sample_psi0 = psi0_all
        self.sample_p0 = p0_all
        
        self.rht_data, self.sample_psi0 = roll_RHT_zero_to_pi(self.rht_data, self.sample_psi0)
        
        try:
            # Add 0.7 because that was the RHT threshold 
            npsample = len(self.sample_p0)
            
            if reverse_RHT is True:
                print("Reversing RHT data")
                self.rht_data = self.rht_data[::-1]
            
            self.prior = (np.array([self.rht_data]*npsample).T + 0.7)*75
            
            self.psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
            self.p_dx = self.sample_p0[1] - self.sample_p0[0]
            
            if self.psi_dx < 0:
                self.psi_dx *= -1
            
            print("psi dx is {}, p dx is {}".format(self.psi_dx, self.p_dx))
            
            self.integrated_over_psi = self.integrate_highest_dimension(self.prior, dx = self.psi_dx)
            self.integrated_over_p_and_psi = self.integrate_highest_dimension(self.integrated_over_psi, dx = self.p_dx)
    
            # Normalize prior over domain
            self.normed_prior = self.prior/self.integrated_over_p_and_psi

        except TypeError:
            if self.rht_data is None:
                print("Index {} not found".format(hp_index))
            else:
                print("Unknown TypeError")
                
                    
class Likelihood(BayesianComponent):
    """
    Class for building Planck-based likelihood
    Currently assumes I = I_0, and sigma_I = 0
    """
    
    def __init__(self, hp_index, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all):
        BayesianComponent.__init__(self, hp_index)      
        (self.hp_index, self.T, self.Q, self.U) = planck_tqu_cursor.execute("SELECT * FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        (self.hp_index, self.TT, self.TQ, self.TU, self.TQa, self.QQ, self.QU, self.TUa, self.QUa, self.UU) = planck_cov_cursor.execute("SELECT * FROM Planck_Nside_2048_cov_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        
        # Naive psi
        self.naive_psi = np.mod(0.5*np.arctan2(self.U, self.Q), np.pi)
        
        # sigma_p as defined in arxiv:1407.0178v1 Eqn 3.
        self.sigma_p = np.zeros((2, 2), np.float_) # [sig_Q^2, sig_QU // sig_QU, UU]
        self.sigma_p[0, 0] = (1.0/self.T**2)*self.QQ #QQ
        self.sigma_p[0, 1] = (1.0/self.T**2)*self.QU #QU
        self.sigma_p[1, 0] = (1.0/self.T**2)*self.QU #QU
        self.sigma_p[1, 1] = (1.0/self.T**2)*self.UU #UU
          
        # det(sigma_p) = sigma_p,G^4
        det_sigma_p = np.linalg.det(self.sigma_p)
        self.sigpGsq = np.sqrt(det_sigma_p)
    
        # measured polarization angle (psi_i = arctan(U_i/Q_i))
        psimeas = np.mod(0.5*np.arctan2(self.U, self.Q), np.pi)

        # measured polarization fraction
        pmeas = np.sqrt(self.Q**2 + self.U**2)/self.T
    
        # invert sigma_p
        invsig = np.linalg.inv(self.sigma_p)
    
        # Sample grid
        nsample = len(p0_all)
        p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))

        # isig array of size (2, 2, nsample*nsample)
        time0 = time.time()
        outfast = np.zeros(nsample*nsample, np.float_)
    
        # Construct measured part
        measpart0 = pmeas*np.cos(2*psimeas)
        measpart1 = pmeas*np.sin(2*psimeas)
    
        p0pairs = p0_psi0_grid[0, ...].ravel()
        psi0pairs = p0_psi0_grid[1, ...].ravel()
    
        # These have length nsample*nsample
        truepart0 = p0pairs*np.cos(2*psi0pairs)
        truepart1 = p0pairs*np.sin(2*psi0pairs)
    
        rharrbig = np.zeros((2, 1, nsample*nsample), np.float_)
        lharrbig = np.zeros((1, 2, nsample*nsample), np.float_)
    
        rharrbig[0, 0, :] = measpart0 - truepart0
        rharrbig[1, 0, :] = measpart1 - truepart1
        lharrbig[0, 0, :] = measpart0 - truepart0
        lharrbig[0, 1, :] = measpart1 - truepart1

        self.likelihood = (1.0/(np.pi*self.sigpGsq))*np.exp(-0.5*np.einsum('ij...,jk...->ik...', lharrbig, np.einsum('ij...,jk...->ik...', invsig, rharrbig)))
        self.likelihood = self.likelihood.reshape(nsample, nsample)

class Posterior(BayesianComponent):
    """
    Class for building a posterior composed of a Planck-based likelihood and an RHT prior
    """
    
    def __init__(self, hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all, reverse_RHT = False):
        BayesianComponent.__init__(self, hp_index)  
        
        # Sample arrays
        self.sample_p0 = p0_all
        self.sample_psi0 = psi0_all
        
        p_dx = p0_all[1] - p0_all[0]
        psi_dx = psi0_all[1] - psi0_all[0]
        
        if psi_dx < 0:
            psi_dx *= -1
        
        self.p_dx = p_dx
        self.psi_dx = psi_dx    
        
        #    psi0_all = psi0_all[::-1]
        #    psi_dx = psi0_all[1] - psi0_all[0]
        #    reverse_RHT = True
        
        print("Posterior psi dx is {}, p dx is {}".format(psi_dx, p_dx))
        self.reverse_RHT = reverse_RHT
        
        # Instantiate posterior components
        prior = Prior(hp_index, rht_cursor, p0_all, psi0_all, reverse_RHT = reverse_RHT)
        likelihood = Likelihood(hp_index, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all)
        
        self.naive_psi = likelihood.naive_psi
        
        self.prior = prior.prior
        self.normed_prior = prior.normed_prior#/np.max(prior.normed_prior)
        self.planck_likelihood = likelihood.likelihood
        
        #self.posterior = np.einsum('ij,jk->ik', self.planck_likelihood, self.normed_prior)
        self.posterior = self.planck_likelihood*self.normed_prior
        
        self.posterior_integrated_over_psi = self.integrate_highest_dimension(self.posterior, dx = psi_dx)
        self.posterior_integrated_over_p_and_psi = self.integrate_highest_dimension(self.posterior_integrated_over_psi, dx = p_dx)
        
        self.normed_posterior = self.posterior/self.posterior_integrated_over_p_and_psi
    
    
if __name__ == "__main__":
    #planck_data_to_database(Nside = 2048, covdata = True)
    #project_allsky_thetaweights_to_database(update = True)
    #reproject_allsky_data()
    
    #project_allsky_singlevel_thetaweights_to_database(update=False, velstr="S0984_0988")
    #write_allsky_singlevel_thetaweights_to_database_RADEC(update = False, velstr="S0984_0988")
    #intRHT_QU_maps_per_vel(velstr="S0974_0978") #haven't done yet
    #intRHT_QU_maps_per_vel(velstr="S0984_0988")
    
    #velstr = "S1069_1073"
    #print('beginning ', velstr)
    #intRHT_QU_maps_per_vel(velstr=velstr)
    
    #write_allsky_singlevel_thetaweights_to_database_RADEC(update = False, velstr=velstr)

    #coadd_QU_maps()
    
    #for _i in np.arange(30, 50):
    #    make_single_theta_int_vel_map(thetabin=_i)
    #reproject_allsky_data()
    
    #c = project_angle0_db(wlen = 75, nest=True)
    
    #QU_RHT_Gal_to_database(smooth=True, sigma=30)
    
    #write_allsky_singlevel_thetaweights_to_database_RADEC_indx(update = False, velstr="S1039_1043")
    
    
    #make_vel_int_galfa_channel_maps()

    #for _i in np.arange(158, 166):
    #    make_weighted_single_theta_int_vel_map(thetabin=_i)
    
    #project_allsky_vel_weighted_int_thetaweights_to_database(update = False)
    
    #reproject_allsky_weighted_data(local=False)
    
    # try loading in already projected data
    #project_allsky_vel_weighted_int_thetaweights_to_database(update=False)
    
    #make_vel_int_galfa_channel_maps()
    
    # make single-vel db indexed by healpix indx, not radec
    project_allsky_singlevel_thetaweights_to_database(update = True, velstr="S0989_0993")
    
