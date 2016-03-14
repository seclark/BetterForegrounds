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

# Local repo imports
import debias

# Other repo imports (RHT helper code)
import sys 
sys.path.insert(0, '../../RHT')
import RHT_tools

"""
 Bayesian psi, p estimation routines.
"""

class BayesianComponent():
    """
    Base class for building Bayesian pieces
    Instantiated by healpix index
    """
    
    def __init__(self, hp_index, verbose = True):
        self.hp_index = hp_index
        self.verbose = verbose
    
    def integrate_highest_dimension(self, field, dx = 1):
        """
        Integrates over highest-dimension axis.
        """
        axis_num = field.ndim - 1
        integrated_field = np.trapz(field, dx = dx, axis = axis_num)
        
        return integrated_field
    
    def get_psi0_sampling_grid(self):
        # Create psi0 sampling grid
        wlen = 75
        psi0_sample_db = sqlite3.connect("theta_bin_0_wlen"+str(wlen)+"_db.sqlite")
        psi0_sample_cursor = psi0_sample_db.cursor()    
        zero_theta = psi0_sample_cursor.execute("SELECT zerotheta FROM theta_bin_0_wlen75 WHERE id = ?", (hp_index,)).fetchone()

        # Create array of projected thetas from theta = 0
        thets = RHT_tools.get_thets(wlen)
        self.sample_psi0 = np.mod(zero_theta - thets, np.pi)
        
        return self.sample_psi0
    
    def roll_RHT_zero_to_pi(self, rht_data, sample_psi):
        # Find index of value closest to 0
        psi_0_indx = np.abs(sample_psi).argmin()
        
        if self.verbose is True:    
            print("rolling data by", psi_0_indx, sample_psi[psi_0_indx])
    
        # Needs 1 extra roll element to be monotonic
        rolled_sample_psi = np.roll(sample_psi, -psi_0_indx - 1)
        rolled_rht = np.roll(rht_data, -psi_0_indx - 1)
        
        return rolled_rht, rolled_sample_psi

class Prior(BayesianComponent):
    """
    Class for building RHT priors
    """
    
    def __init__(self, hp_index, reverse_RHT = False):
    
        BayesianComponent.__init__(self, hp_index)
        
        # Planck-projected RHT database
        rht_db = sqlite3.connect("allweights_db.sqlite")
        rht_cursor = rht_db.cursor()
        self.rht_data = rht_cursor.execute("SELECT * FROM RHT_weights WHERE id = ?", (self.hp_index,)).fetchone()
        
        # Discard first element because it is the healpix id
        self.rht_data = self.rht_data[1:]
        
        # Get sample psi data
        self.sample_psi0 = self.get_psi0_sampling_grid()
        
        # Roll RHT data to [0, pi)
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
    
    def __init__(self, hp_index, rht_cursor, planck_tqu_cursor, planck_cov_cursor):
        BayesianComponent.__init__(self, hp_index)  
        
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
    