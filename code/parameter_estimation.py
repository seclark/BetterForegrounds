from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time

import debias

"""
 Simple psi, p estimation routines.
"""

def ax_posterior(ax, p0s, psi0s, B2D, cmap = "hsv"):
    """
    Plot 2D posterior distribution on given axes instance
    """
    
    obj = ax.pcolormesh(p0s, psi0s, B2D, cmap = cmap)
    ax.set_xlim(np.nanmin(p0s), np.nanmax(p0s))
    ax.set_ylim(np.nanmin(psi0s), np.nanmax(psi0s))
    ax.set_xlabel(r"$p_0$", size = 15)
    ax.set_ylabel(r"$\psi_0$", size = 15)
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
    Nex = 5
    
    # eps = 1, rho = 0, (eps_eff = 1, theta = 0)
    eps = np.asarray([1.0, 0.5, 2.0, 1.0, 1.0])
    rho = np.asarray([0, 0, 0, -0.5, 1.5])
    
    for i in xrange(Nex):
        covmatrix = np.zeros((2, 2, Nex), np.float_)
        covmatrix[0, 0, i] = eps[i]
        covmatrix[0, 1, i] = rho[i]
        covmatrix[1, 0, i] = rho[i]
        covmatrix[1, 1, i] = 1.0/eps[i]
        
    sigma_p = covmatrix

    psimeas = np.repeat(0.0, Nex)
    pmeas = np.repeat(0.1, Nex)
    
    invsig = np.linalg.inv(sigma_p.swapaxes(0, 2))
    
    print("Done inverting sigma_p")

    # Create grid of psi0's and p0's to sample
    nsample = 100
    psi0_all = np.linspace(0, np.pi, nsample)
    p0_all = np.linspace(0, 1.0, nsample)

    p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))
    
    print("starting slow way")
    time0 = time.time()
    out = np.zeros((Npix, nsample*nsample), np.float_) 
    for p, isig in enumerate(invsig[:Npix, :, :]): #the :Npix is a hack - should go away

        for (i, (p0, psi0)) in enumerate(p0_psi0_pairs):

            # Implement Montier+ II Eq. 25
            rharr[0, 0] = pmeas[p]*np.cos(2*psimeas[p]) - p0*np.cos(2*psi0)
            rharr[1, 0] = pmeas[p]*np.sin(2*psimeas[p]) - p0*np.sin(2*psi0)

            # Lefthand array is transpose of righthand array
            lharr = rharr.T
    
            out[p, i] = (1/(np.pi*sigpGsq[p]))*np.exp(-0.5*np.dot(lharr, np.dot(isig, rharr)))
    time1 = time.time()
    print("process took ", time1 - time0, "seconds")
    
    return out

def Planck_posteriors(map353Gal = None, cov353Gal = None):
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

    # temporary hack -- only look at first 1000 points
    Npix = 1000
    sigma_p = sigma_p[:, :, 0:Npix]
    
    # invert matrix -- must have Npix axis first
    invsig = np.linalg.inv(sigma_p.swapaxes(0, 2))
    
    print("Done inverting sigma_p")

    # Create grid of psi0's and p0's to sample
    nsample = 100
    psi0_all = np.linspace(0, np.pi, nsample)
    p0_all = np.linspace(0, 1.0, nsample)

    p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))
    """
    p0_psi0_pairs = zip(p0_psi0_grid[0, ...].ravel(), p0_psi0_grid[1, ...].ravel())

    rharr = np.zeros((2, 1), np.float_)
    
    # Brute force loop first
    print("starting slow way")
    time0 = time.time()
    out = np.zeros((Npix, nsample*nsample), np.float_) 
    for p, isig in enumerate(invsig[:Npix, :, :]): #the :Npix is a hack - should go away

        for (i, (p0, psi0)) in enumerate(p0_psi0_pairs):

            # Implement Montier+ II Eq. 25
            rharr[0, 0] = pmeas[p]*np.cos(2*psimeas[p]) - p0*np.cos(2*psi0)
            rharr[1, 0] = pmeas[p]*np.sin(2*psimeas[p]) - p0*np.sin(2*psi0)

            # Lefthand array is transpose of righthand array
            lharr = rharr.T
    
            out[p, i] = (1/(np.pi*sigpGsq[p]))*np.exp(-0.5*np.dot(lharr, np.dot(isig, rharr)))
    time1 = time.time()
    print("process took ", time1 - time0, "seconds")
    """
    # Testing new "fast way" that works for isig array of size (2, 2, nsample*nsample) s.t. loop is over Npix
    print("starting fast way")
    outfast = np.zeros((Npix, nsample*nsample), np.float_)
    time0 = time.time()
    
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
    
    
    # Is this faster?
    #uu = np.einsum('ij, jlk -> ilk', lharr, np.einsum('ijk, jl -> ilk', sp100, rharr)) # single
    
    # this is for all Npoints p... so just make rharr and lharr incl all (p0, psi0) and will be good.
    # Will work for isig array shape (2, 2, Npix) and rharrbig shape (2, 1, Npix)
    """
    print("starting fast way")
    outfast = np.zeros((Npix, nsample*nsample), np.float_)
    time0 = time.time()
    measpart0 = pmeas*np.cos(2*psimeas)
    measpart1 = pmeas*np.sin(2*psimeas)
    
    p0pairs = p0_psi0_grid[0, ...].ravel()
    psi0pairs = p0_psi0_grid[1, ...].ravel()
    
    truepart0 = p0pairs*np.cos(2*psi0pairs)
    truepart1 = p0pairs*np.sin(2*psi0pairs)
    
    measpart0 = measpart0.reshape(len(measpart0), 1)
    measpart1 = measpart1.reshape(len(measpart1), 1)
    truepart0 = truepart0.reshape(1, len(truepart0))
    truepart1 = truepart1.reshape(1, len(truepart1))
    
    # Too memory intensive
    rharrbig = np.zeros((2, 1, Npix, nsample*nsample), np.float_)
    lharrbig = np.zeros((1, 2, Npix, nsample*nsample), np.float_)
    
    rharrbig[0, 0, :, :] = measpart0 - truepart0
    rharrbig[1, 0, :, :] = measpart1 - truepart1
    
    lharrbig[0, 0, :, :] = measpart0 - truepart0
    lharrbig[0, 1, :, :] = measpart1 - truepart1
    
    print("entering loop")
    for i, (p0, psi0) in enumerate(p0_psi0_pairs):
        outfast[:, i] = np.einsum('ij...,jk...->ik...', lharrbig[:, :, :, i], np.einsum('ij...,jk...->ik...', isig, rharrbig[:, :, :, i]))
    time1 = time.time()
    """
    """
    time0 = time.time()
    # This will work for isig array shape (2, 2, Npix, nsample*nsample). 
    isigbig = np.repeat(isig[:, :, np.newaxis], Npix, axis=2)
    isigbig = np.repeat(isigbig[:, :, :, np.newaxis], nsample*nsample, axis=3)
    print(isigbig.shape)
    
    outfast = np.einsum('ij...,jk...->ik...', lharrbig, np.einsum('ij...,jk...->ik...', isigbig, rharrbig))
    time1 = time.time()
    """
    print("fast version took ", time1 - time0, "seconds")
    
    return outfast
    
    
        
#outfast = Planck_posteriors()
