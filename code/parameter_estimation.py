from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time

import debias

"""
 Simple psi, p estimation routines.
"""

full_planck_fn = "/Users/susanclark/Dropbox/GALFA-Planck/Big_Files/HFI_SkyMap_353_2048_R2.02_full.fits"

# resolution
Nside=2048
Npix = 12*Nside**2

# input Planck 353 GHz maps (Galactic)
# full-mission -- N.B. these maps are already in RING ordering, despite what the header says
map353Gal = np.zeros((3,Npix)) #T,Q,U
cov353Gal = np.zeros((3,3,Npix)) #TT,TQ,TU,QQ,QU,UU
map353Gal[0], map353Gal[1], map353Gal[2], cov353Gal[0,0], cov353Gal[0,1], cov353Gal[0,2], cov353Gal[1,1], cov353Gal[1,2], cov353Gal[2,2], header353Gal = hp.fitsfunc.read_map(full_planck_fn, field=(0,1,2,4,5,6,7,8,9), h=True)


# sigma_p as defined in arxiv:1407.0178v1 Eqn 3.
sigma_p = np.zeros((2, 2, Npix)) # [sig_Q^2, sig_QU // sig_QU, UU]
sigma_p[0, 0, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 1, :] #QQ
sigma_p[0, 1, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 2, :] #QU
sigma_p[1, 0, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[1, 2, :] #QU
sigma_p[1, 1, :] = (1.0/map353Gal[0, :]**2)*cov353Gal[2, 2, :] # UU

# Montier+ II Eq. 5
#eps = np.sqrt(cov353Gal[1, 1, :]/cov353Gal[2, 2, :])
#rhosq = cov353Gal[1, 2, :]/(cov353Gal[1, 1, :]*cov353Gal[2, 2, :])
#sigpGsq = (cov353Gal[1, 1, :]/map353Gal[0, :]**2)*(np.sqrt(1 - rhosq)/eps)

# Assume rho = 1, so define det(sigma_p) = sigma_p,G^4
det_sigma_p = np.linalg.det(sigma_p.swapaxes(0, 2))
sigpGsq = np.sqrt(det_sigma_p)

# measured polarization angle (psi_i = arctan(U_i/Q_i))
psimeas = np.mod(0.5*np.arctan2(map353Gal[2, :], map353Gal[1, :]), np.pi)

# measured polarization fraction
pmeas = np.sqrt(map353Gal[1, :]**2 + map353Gal[2, :]**2)

# invert matrix -- must have Npix axis first
invsig = np.linalg.inv(sigma_p.swapaxes(0, 2))
#invsig = invsig.swapaxes(0, 2)

# Create grid of psi0's and p0's to sample
nsample = 100
psi0_all = np.linspace(0, np.pi, nsample)
p0_all = np.linspace(0, 1.0, nsample)

p0_psi0_grid = np.asarray(np.meshgrid(p0_all, psi0_all))
p0_psi0_pairs = zip(p0_psi0_grid[0, ...].ravel(), p0_psi0_grid[1, ...].ravel())

rharr = np.zeros((2, 1), np.float_)
# temporary hack -- only look at first 1000 points
Npix = 1000

# Brute force loop first
time0 = time.time()
out = np.zeros((Npix, nsample*nsample), np.float_) # 10 -> Npix in real thing, just calculating first 10 for now
for p, isig in enumerate(invsig):

    for (i, (p0, psi0)) in enumerate(p0_psi0_pairs):

        # Implement Montier+ II Eq. 25
        rharr[0, 0] = pmeas[p]*np.cos(2*psimeas[p]) - p0*np.cos(2*psi0)
        rharr[1, 0] = pmeas[p]*np.sin(2*psimeas[p]) - p0*np.sin(2*psi0)
    
        # Lefthand array is transpose of righthand array
        lharr = rharr.T
        
        out[p, i] = (1/(np.pi*sigpGsq[p]))*np.exp(0.5*np.dot(lharr, np.dot(isig, rharr)))
time1 = time.time()
print("process took ", time1 - time0, "seconds")
        
        
        
    
    
    
