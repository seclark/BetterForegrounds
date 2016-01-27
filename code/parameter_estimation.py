from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
from astropy.io import fits

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
    
    #obj = ax.pcolormesh(p0s, psi0s, B2D, cmap = cmap)
    maxval = np.nanmax(B2D)
    levels = np.asarray([0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9])*maxval
    obj = ax.contourf(p0s, psi0s, B2D, levels, cmap = cmap, extend = "both")
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
            ax.plot([pmeas[indx], pmeas[indx]], [0, np.pi], '--', color = "pink", lw = 3)
            ax.plot([0, 1], [psimeas[indx], psimeas[indx]], '--', color = "pink", lw = 3)
            
        if sigpGsq != None:
            ax.set_title(r"$"+str(indx)+",$ $p_{meas}/\sigma_{p, G} = "+str(np.round(pmeas[indx]/sigpGsq[indx], 1))+"$", size = 15)

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
                ax.plot([pmeas[i], pmeas[i]], [-np.pi/2.0, np.pi/2.0], '--', color = "pink", lw = 3)
            else:                
                ax.plot([pmeas[i], pmeas[i]], [0, np.pi], '--', color = "pink", lw = 3)
            ax.plot([0, 1], [psimeas[i], psimeas[i]], '--', color = "pink", lw = 3)
            
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

def get_Planck_data_projected(Nside = 2048, region = "SC_241"):

    fn_root = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/" 
    cov_fn = fn_root + "HFI_SkyMap_353_"+str(Nside)+"_R2.02_full_IAU_"+region+"_projected_cov.fits"
    TQU_fn = fn_root + "HFI_SkyMap_353_"+str(Nside)+"_R2.02_full_IAU_"+region+"_projected_TQUonly.fits"
    
    TQUcov = fits.getdata(cov_fn)
    TQU = fits.getdata(TQU_fn)
    
    return TQUcov, TQU

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

            # Implement Montier+ II Eq. 25
            rharr[0, 0] = pmeas[p]*np.cos(2*psimeas[p]) - p0*np.cos(2*psi0)
            rharr[1, 0] = pmeas[p]*np.sin(2*psimeas[p]) - p0*np.sin(2*psi0)

            # Lefthand array is transpose of righthand array
            lharr = rharr.T
    
            out[p, i] = (1/(np.pi*np.sqrt(sig_pG[p])))*np.exp(-0.5*np.dot(lharr, np.dot(isig, rharr)))
    time1 = time.time()
    print("process took ", time1 - time0, "seconds")
    
    plot_test_posteriors(pmeas, psimeas, p0_all, psi0_all, out, cmap = "jet")
    
    return out, covmatrix

def Planck_posteriors(map353Gal = None, cov353Gal = None, firstnpoints = None):
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
    
        outfast[i, :] = np.einsum('ij...,jk...->ik...', lharrbig, np.einsum('ij...,jk...->ik...', invsig[i, :, :], rharrbig))
    time1 = time.time()
    print("fast version took ", time1 - time0, "seconds")
    
    return outfast

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
    if map353Gal == None:
        map353Gal, cov353Gal = get_Planck_data(Nside = Nside)

    # likelihood = planck-only posterior
    likelihood = Planck_posteriors(map353Gal = map353Gal, cov353Gal = cov353Gal, firstnpoints = firstnpoints)

        
def add_hthets(data1, data2):
    """
    Combine two R(theta) arrays
    Overlapping (x, y) points have associated R(theta) arrays summed
    Unique (x, y) points are added to the list
    """

    for key in data2.keys():
        if data1.has_key(key):
            data1[key] = [data1[key]] + [data2[key]]
            data1[key] = list(np.sum(data1[key], axis=0))
        else:
            data1[key] = data2[key]
    
    return data1
    
    
# Pull in each projected theta bin
projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"

# resolution
Nside = 2048
Npix = 12*Nside**2

# These are the healpix indices. They will be the keys in our dictionary.
hp_index = np.arange(Npix)

nthets = 165 


for _thetabin_i in xrange(nthets):
    projected_fn = projected_root + "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_"+str(_thetabin_i)+".fits"
    projdata = fits.getdata(projected_fn)
    
    # Some data stored as -999 for 'none'
    projdata[projdata == -999] = 0
    
    nonzero_indx = np.nonzero(projdata)[0]
    print("there are {} nonzero elements").format(len(nonzero_indx))

    
jitot = {}
if len(ch) > 1:
    for i in xrange(len(ch)):
        print "loading channel %d" % (ch[i])
        
         projected_root = "/Volumes/DataDavy/GALFA/SC_241/cleaned/galfapix_corrected/theta_backprojections/"
    projected_fn = "SC_241.66_28.675.best_16_24_w75_s15_t70_galfapixcorr_thetabin_99.fits"
    
        ipoints, jpoints, hthets, naxis1, naxis2 = RHT_tools.get_RHT_data(xyt_filename = xyt_fn)
        
        jipoints2 = zip(jpoints2, ipoints2)
        jih2 = dict(zip(jipoints2, hthets2))
        
        jitot = add_hthets(jitot, jih2)
        


    
    