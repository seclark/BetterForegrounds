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

# Local repo imports
import debias

# Other repo imports (RHT helper code)
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

"""
 Simple psi, p estimation routines.
"""

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
    
def project_angles_db(wlen = 75):
    """
    Project angles from Equatorial, B-field, IAU Definition -> Galactic, Polarization Angle, Planck Definition
    Store in SQL Database by healpix id
    Note: projected angles are still equally spaced -- no need to re-interpolate
    """
    
    zero_thetas = fits.getdata("/Volumes/DataDavy/Planck/projected_angles/theta_0.0_Equ_inGal.fits")
    thets = RHT_tools.get_thets(wlen)
    nthets = len(thets)
    
    # Arbitrary 2-letter SQL storage value names
    value_names = [''.join(i) for i in itertools.permutations(string.lowercase,2)]

    # Remove protected words from value names
    if "as" in value_names: value_names.remove("as")
    if "is" in value_names: value_names.remove("is")

    # Comma separated list of nthets column names
    column_names = " FLOAT DEFAULT 0.0,".join(value_names[:nthets])

    # Name table
    tablename = "theta_bins_wlen"+str(wlen)

    # Statement for creation of SQL database
    createstatement = "CREATE TABLE "+tablename+" (id INTEGER PRIMARY KEY,"+column_names+" FLOAT DEFAULT 0.0);"

    # Instantiate database
    conn = sqlite3.connect(":memory:")
    #conn = sqlite3.connect("theta_bins_wlen75_db.sqlite")
    c = conn.cursor()
    c.execute(createstatement)
    conn.commit()
    
    insertstatement = "INSERT INTO "+tablename+" VALUES ("+",".join('?'*nthets)+")"
    
    Npix = 10
    # One-liner that == Colin's loop in rht_to_planck.py
    thets_EquinGal = np.mod(np.asarray(zero_thetas[:Npix]).reshape(Npix, 1).astype(np.float_) - thets, np.pi)
    
    for _hp_index in xrange(Npix):
        thets_EquinGal = np.mod(np.asarray(zero_thetas[_hpindex]).reshape(nthets, 1).astype(np.float_) - thets, np.pi)
        c.execute(insertstatement, itertools.chain([_hp_index], thets_EquinGal))    
    
    return thets_EquinGal
        
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
        integrated_field = np.trapz(field, dx = dx)
        
        return integrated_field
    

class Prior(BayesianComponent):
    """
    Class for building RHT priors
    """
    
    def __init__(self, hp_index, c, psibins = None, npsample = 165, npsisample = 165):
    
        BayesianComponent.__init__(self, hp_index)
        self.rht_data = c.execute("SELECT * FROM RHT_weights WHERE id = ?", (self.hp_index,)).fetchone()
        
        try:
            # Add 0.7 because that was the RHT threshold 
            self.prior = np.array([self.rht_data[1:]]*npsample).T + 0.7
            
            self.integrated_over_psi = self.integrate_highest_dimension(self.prior, dx = np.pi/npsisample)
            self.integrated_over_p_and_psi = self.integrate_highest_dimension(self.integrated_over_psi, dx = 1.0/npsample)
    
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
    """
    
    def __init__(self, hp_index, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all):
        BayesianComponent.__init__(self, hp_index)      
        (self.hp_index, self.T, self.Q, self.U) = planck_tqu_cursor.execute("SELECT * FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        (self.hp_index, self.TT, self.TQ, self.TU, self.TQa, self.QQ, self.QU, self.TUa, self.QUa, self.UU) = planck_cov_cursor.execute("SELECT * FROM Planck_Nside_2048_cov_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        
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

        self.likelihood = (1/(np.pi*self.sigpGsq))*np.exp(-0.5*np.einsum('ij...,jk...->ik...', lharrbig, np.einsum('ij...,jk...->ik...', invsig, rharrbig)))

class Posterior(BayesianComponent):
    """
    Class for building a posterior composed of a Planck-based likelihood and an RHT prior
    """
    
    def __init__(self, hp_index):
        BayesianComponent.__init__(self, hp_index)  
    
#if __name__ == "__main__":
#    planck_data_to_database(Nside = 2048, covdata = True)


