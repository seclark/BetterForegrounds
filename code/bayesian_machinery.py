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
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc
rc('text', usetex=True)

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
    
    def get_psi0_sampling_grid(self, hp_index):
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
    
    def __init__(self, hp_index, sample_p0, reverse_RHT = False):
    
        BayesianComponent.__init__(self, hp_index)
        
        # Planck-projected RHT database
        rht_db = sqlite3.connect("allweights_db.sqlite")
        rht_cursor = rht_db.cursor()
        self.rht_data = rht_cursor.execute("SELECT * FROM RHT_weights WHERE id = ?", (self.hp_index,)).fetchone()
        
        self.sample_p0 = sample_p0
        
        try:
            # Discard first element because it is the healpix id
            self.rht_data = self.rht_data[1:]
        
            # Get sample psi data
            self.sample_psi0 = self.get_psi0_sampling_grid(hp_index)
        
            # Roll RHT data to [0, pi)
            self.rht_data, self.sample_psi0 = self.roll_RHT_zero_to_pi(self.rht_data, self.sample_psi0)
        
            # Add 0.7 because that was the RHT threshold 
            npsample = len(self.sample_p0)
            
            if reverse_RHT is True:
                print("Reversing RHT data")
                self.rht_data = self.rht_data[::-1]
                self.sample_psi0 = self.sample_psi0[::-1]
            
            self.prior = (np.array([self.rht_data]*npsample).T + 0.7)*75
            
            self.psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
            self.p_dx = self.sample_p0[1] - self.sample_p0[0]
            
            if self.psi_dx < 0:
                print("Multiplying psi_dx by -1")
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
                print("Unknown TypeError when constructing RHT prior")
                
                    
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
    
    def __init__(self, hp_index, sample_p0 = None):
        BayesianComponent.__init__(self, hp_index)  
        
        if sample_p0 is None:
            self.sample_p0 = np.linspace(0, 1, 165)
        else:
            self.sample_p0 = sample_p0
        
        # Instantiate posterior components
        prior = Prior(hp_index, self.sample_p0, reverse_RHT = True)
        self.sample_psi0 = prior.sample_psi0
        
        # Planck covariance database
        planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
        planck_cov_cursor = planck_cov_db.cursor()
    
        # Planck TQU database
        planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
        planck_tqu_cursor = planck_tqu_db.cursor()
        
        # Planck-based likelihood
        likelihood = Likelihood(hp_index, planck_tqu_cursor, planck_cov_cursor, self.sample_p0, self.sample_psi0)
        
        self.naive_psi = likelihood.naive_psi
        
        self.normed_prior = prior.normed_prior#/np.max(prior.normed_prior)
        self.planck_likelihood = likelihood.likelihood
        
        #self.posterior = np.einsum('ij,jk->ik', self.planck_likelihood, self.normed_prior)
        self.posterior = self.planck_likelihood*self.normed_prior
        
        psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
        p_dx = self.sample_p0[1] - self.sample_p0[0]
        
        self.posterior_integrated_over_psi = self.integrate_highest_dimension(self.posterior, dx = psi_dx)
        self.posterior_integrated_over_p_and_psi = self.integrate_highest_dimension(self.posterior_integrated_over_psi, dx = p_dx)
        
        self.normed_posterior = self.posterior/self.posterior_integrated_over_p_and_psi

def latex_formatter(x, pos):
    return "${0:.1f}$".format(x)

def plot_bayesian_component_from_posterior(posterior_obj, component = "posterior", ax = None, cmap = "cubehelix"):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    extent = [posterior_obj.sample_p0[0], posterior_obj.sample_p0[-1], posterior_obj.sample_psi0[0], posterior_obj.sample_psi0[-1]]
    
    aspect = (posterior_obj.sample_p0[1] - posterior_obj.sample_p0[0])/(posterior_obj.sample_psi0[1] - posterior_obj.sample_psi0[0])
    aspect2 = 0.15*(posterior_obj.sample_p0[1] - posterior_obj.sample_p0[0])/(posterior_obj.sample_psi0[1] - posterior_obj.sample_psi0[0])
    
    if component == "posterior":
        plotarr = posterior_obj.normed_posterior
        title = r"$\mathrm{Posterior}$"
    if component == "likelihood":
        plotarr = posterior_obj.planck_likelihood
        title = r"$\mathrm{Planck}$ $\mathrm{Likelihood}$"
    if component == "prior":
        plotarr = posterior_obj.normed_prior
        title = r"$\mathrm{RHT}$ $\mathrm{Prior}$"
    
    im = ax.imshow(plotarr, cmap = cmap, extent = extent, aspect = aspect)
    
    ax.set_title(title, size = 20)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="15%", pad=0.05, aspect = 100./15)
    cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(latex_formatter))
    
def plot_all_bayesian_components_from_posterior(posterior_obj, cmap = "cubehelix"):
    
    fig = plt.figure(figsize = (10, 4), facecolor = "white")
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    plot_bayesian_component_from_posterior(posterior_obj, component = "likelihood", ax = ax1, cmap = cmap)
    plot_bayesian_component_from_posterior(posterior_obj, component = "prior", ax = ax2, cmap = cmap)
    plot_bayesian_component_from_posterior(posterior_obj, component = "posterior", ax = ax3, cmap = cmap)
    
    plt.subplots_adjust(wspace = 0.8)
    
    pMB, psiMB = mean_bayesian_posterior(posterior_obj)
    ax3.plot(pMB, psiMB, '+', ms = 20, color = "white")
    
    pnaive, psinaive = naive_planck_measurements(posterior_obj.hp_index)
    ax1.plot(pnaive, psinaive, '+', ms = 20, color = "white")
    
    axs = [ax1, ax2, ax3]
    for ax in axs:
        ax.set_ylim(posterior_obj.sample_psi0[0], posterior_obj.sample_psi0[-1])
    
def naive_planck_measurements(hp_index):
    
    # Planck TQU database
    planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
    planck_tqu_cursor = planck_tqu_db.cursor()
    
    # Get I0, Q0, U0 measurements
    I0 = planck_tqu_cursor.execute("SELECT T FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    Qmeas = planck_tqu_cursor.execute("SELECT Q FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    Umeas = planck_tqu_cursor.execute("SELECT U FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (hp_index,)).fetchone()
    
    Pnaive = np.sqrt(Qmeas[0]**2 + Umeas[0]**2)
    pnaive = Pnaive/I0
    psinaive = np.mod(0.5*np.arctan2(Umeas, Qmeas), np.pi)

    print("Naive p is {}".format(pnaive))
    print("Naive psi is {}".format(psinaive))
    
    return pnaive, psinaive
    
def center_naive_measurements(hp_index, sample_p0, center_on_p, sample_psi0, center_on_psi):

    pnaive, psinaive = naive_planck_measurements(hp_index)

    # Find index of value closest to pnaive and psinaive
    pnaive_indx = np.abs(sample_p0 - pnaive).argmin()
    psinaive_indx = np.abs(sample_psi0 - psinaive).argmin()
    
    rolled_sample_p0 = np.roll(sample_p0, - pnaive_indx)
    rolled_weights_p0 = np.roll(center_on_p, - pnaive_indx)
    
    rolled_sample_psi0 = np.roll(sample_psi0, - psinaive_indx)
    rolled_weights_psi0 = np.roll(center_on_psi, - psinaive_indx)
    
    return rolled_sample_p0, rolled_weights_p0, rolled_sample_psi0, rolled_weights_psi0
    
def center_posterior_naive_psi(hp_index, sample_psi0, posterior):

    pnaive, psinaive = naive_planck_measurements(hp_index)

    # Find index of value closest to pnaive and psinaive
    psinaive_indx = np.abs(sample_psi0 - (psinaive - np.pi/2)).argmin()
    
    rolled_posterior = np.roll(posterior, - psinaive_indx, axis = 0)
    
    rolled_sample_psi0 = np.roll(sample_psi0, - psinaive_indx)
    rolled_sample_psi0[rolled_sample_psi0 < psinaive - np.pi/2] += np.pi
    
    return rolled_sample_psi0, rolled_posterior 
    
def mean_bayesian_posterior(posterior_obj):
    """
    Integrated first order moments of the posterior PDF
    """
    
    posterior = posterior_obj.normed_posterior
    
    sample_p0 = posterior_obj.sample_p0
    sample_psi0 = posterior_obj.sample_psi0
    
    grid_sample_p0 = np.tile(sample_p0, (len(sample_p0), 1))
    grid_sample_psi0 = np.tile(np.reshape(sample_psi0[::-1], (len(sample_psi0), 1)), (1, len(sample_psi0)))
    
    # First moment of p0 map
    p0moment1 = grid_sample_p0*posterior
    
    # Sampling widths
    pdx = sample_p0[1] - sample_p0[0]
    psidx = sample_psi0[1] - sample_psi0[0]
    
    # Test that normed posterior is normed
    norm_posterior_test = test_normalization(posterior_obj, pdx, psidx)
    
    print("Sampling pdx is {}, psidx is {}".format(pdx, psidx))
    
    # Axis 0 integrates over psi
    
    # Test
    rolled_sample_psi0, rolled_posterior = center_posterior_naive_psi(posterior_obj.hp_index, sample_psi0, posterior)
    posterior = rolled_posterior
    sample_psi0 = rolled_sample_psi0
    
    
    # Integrate over p
    #pMB1 = np.trapz(p0moment1, dx = pdx, axis = 0)
    pMB1 = np.trapz(posterior, dx = psidx, axis = 0)
    
    # Integrate over psi
    #pMB = np.trapz(pMB1, dx = psidx)
    pMB = np.trapz(pMB1*sample_p0, dx = pdx)
    
    # First moment of psi0 map
    psi0moment1 = grid_sample_psi0*posterior
    
    # Normalize first moment map
    #integrated_psi0moment1 = np.trapz(psi0moment1, dx = psidx, axis = 1)
    #integrated_psi0moment1 = np.trapz(integrated_psi0moment1, dx = pdx, axis = 0)
    #print(integrated_psi0moment1)
    #psi0moment1 = psi0moment1/integrated_psi0moment1
    
    # Integrate over p
    #psiMB1 = np.trapz(psi0moment1, dx = pdx, axis = 0)
    psiMB1 = np.trapz(posterior, dx = pdx, axis = 1)
    
    #test
    #rolled_sample_p0, rolled_pMB1, rolled_sample_psi0, rolled_psiMB1 = center_naive_measurements(posterior_obj.hp_index, sample_p0, pMB1, sample_psi0, psiMB1)
    #pMB = np.trapz(rolled_pMB1*rolled_sample_p0, dx = pdx) #test
    
    # Integrate over psi
    #psiMB = np.trapz(psiMB1, dx = psidx)
    psiMB = np.trapz(psiMB1*sample_psi0, dx = psidx)
    
    #psiMB = np.trapz(rolled_psiMB1*rolled_sample_psi0, dx = psidx) #test
    
    print("pMB is {}".format(pMB))
    print("psiMB is {}".format(psiMB))
    
    return pMB, psiMB#, pMB1, psiMB1, sample_psi0, sample_p0

def test_normalization(posterior_obj, pdx, psidx):
    norm_posterior_test = posterior_obj.integrate_highest_dimension(posterior_obj.normed_posterior, dx = psidx)
    norm_posterior_test = posterior_obj.integrate_highest_dimension(norm_posterior_test, dx = pdx)
    
    print("Normalized posterior is {}".format(norm_posterior_test))
    
    return norm_posterior_test
    
    