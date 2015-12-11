from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite

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
sigma_p[0, 0, :] = (1.0/map353Gal[0]**2)*cov353Gal[1, 1, :] #QQ
sigma_p[0, 1, :] = (1.0/map353Gal[0]**2)*np.sqrt(cov353Gal[1, 2, :]) #QU
sigma_p[1, 0, :] = (1.0/map353Gal[0]**2)*np.sqrt(cov353Gal[1, 2, :]) #QU
sigma_p[1, 1, :] = (1.0/map353Gal[0]**2)*cov353Gal[2, 2, :] # UU

# measured polarization angle (phi_i = arctan(U_i/Q_i))
mapphi353Gal = np.mod(0.5*np.arctan2(map353Gal[2], map353Gal[1]), np.pi)

# invert matrix -- must have Npix axis first
invsig = np.linalg.inv(sigma_p.swapaxes(0, 2))
invsig = invsig.swapaxes(0, 2)


