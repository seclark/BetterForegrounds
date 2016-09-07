from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import cPickle as pickle
import itertools
import string
import sqlite3
import scipy
from scipy import special
import scipy.ndimage
import copy
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

sys.path.insert(0, '../../PolarizationTools')
import basic_functions as polarization_tools

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
    
    def get_psi0_sampling_grid(self, hp_index, verbose = True):
        # Create psi0 sampling grid
        wlen = 75
        psi0_sample_db = sqlite3.connect("theta_bin_0_wlen"+str(wlen)+"_db.sqlite")
        psi0_sample_cursor = psi0_sample_db.cursor()    
        zero_theta = psi0_sample_cursor.execute("SELECT zerotheta FROM theta_bin_0_wlen75 WHERE id = ?", (hp_index,)).fetchone()

        # Create array of projected thetas from theta = 0
        thets = RHT_tools.get_thets(wlen, save = False, verbose = verbose)
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
        
    def get_adaptive_p_grid(self, hp_index):
        # Planck TQU database
        planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
        planck_tqu_cursor = planck_tqu_db.cursor()
    
        (self.hp_index, self.T, self.Q, self.U) = planck_tqu_cursor.execute("SELECT * FROM Planck_Nside_2048_TQU_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        
        pmeas = np.sqrt(self.Q**2 + self.U**2)/self.T
        
        # Planck covariance database
        planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
        planck_cov_cursor = planck_cov_db.cursor()
        
        (self.hp_index, self.TT, self.TQ, self.TU, self.TQa, self.QQ, self.QU, self.TUa, self.QUa, self.UU) = planck_cov_cursor.execute("SELECT * FROM Planck_Nside_2048_cov_Galactic WHERE id = ?", (self.hp_index,)).fetchone()
        
        # from Planck Intermediate Results XIX eq. B.2. Taking I0 to be perfectly known
        sigpsq = (1/(pmeas**2*self.T**4))*(self.Q**2*self.QQ + self.U**2*self.UU + 2*self.Q*self.U*self.QU)
        sigmameas = np.sqrt(sigpsq)
        
        # grid bounded at +/- 1
        pgridmin = max(0, pmeas - 7*sigmameas)
        pgridmax = min(1, pmeas + 7*sigmameas)
        
        # grid must be centered on p0
        mindist = min(pmeas - pgridmin, pgridmax - pmeas)
        
        pgridstart = pmeas - mindist
        pgridstop = pmeas + mindist
        
        pgrid = np.linspace(pgridstart, pgridstop, 165)
        
        #diagnostics
        #print("naive p = {}, sigma = {}, therefore bounds are p = {} to {}".format(pmeas, sigmameas, pgridstart, pgridstop))
        if pgridstart < 0.0:
            print("CAUTION: pgridstart = {} for index {}".format(pgridstart, hp_index))
        if pgridstop > 1.0:
            print("CAUTION: pgridstop = {} for index {}".format(pgridstop, hp_index))
        
        return pgrid
        

class Prior(BayesianComponent):
    """
    Class for building RHT priors
    """
    
    def __init__(self, hp_index, sample_p0, reverse_RHT = False, verbose = False, region = "SC_241", rht_cursor = None, gausssmooth = False):
    
        BayesianComponent.__init__(self, hp_index, verbose = verbose)
        
        # Planck-projected RHT database
        #rht_cursor, tablename = get_rht_cursor(region = region)
        
        if region is "allsky":
            self.rht_data = rht_cursor.execute("SELECT * FROM RHT_weights_allsky WHERE id = ?", (self.hp_index,)).fetchone()
        if region is "SC_241":
            self.rht_data = rht_cursor.execute("SELECT * FROM RHT_weights WHERE id = ?", (self.hp_index,)).fetchone()
        
        self.sample_p0 = sample_p0
        
        try:
            # Discard first element because it is the healpix id
            self.rht_data = self.rht_data[1:]
            
            if gausssmooth is True:
                # Gaussian smooth with sigma = 3, wrapped boundaries for filter
                self.rht_data = scipy.ndimage.gaussian_filter1d(self.rht_data, 3, mode = "wrap")
        
            # Get sample psi data
            self.sample_psi0 = self.get_psi0_sampling_grid(hp_index, verbose = verbose)
        
            # Roll RHT data to [0, pi)
            self.rht_data, self.sample_psi0 = self.roll_RHT_zero_to_pi(self.rht_data, self.sample_psi0)
        
            # Add 0.7 because that was the RHT threshold 
            npsample = len(self.sample_p0)
            
            if reverse_RHT is True:
                if verbose is True:
                    print("Reversing RHT data")
                self.rht_data = self.rht_data[::-1]
                self.sample_psi0 = self.sample_psi0[::-1]
            
            self.prior = (np.array([self.rht_data]*npsample).T + 0.7)*75
            
            self.psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
            self.p_dx = self.sample_p0[1] - self.sample_p0[0]
            
            if self.psi_dx < 0:
                print("Multiplying psi_dx by -1")
                self.psi_dx *= -1
            
            if verbose is True:
                print("psi dx is {}, p dx is {}".format(self.psi_dx, self.p_dx))
            
            self.integrated_over_psi = self.integrate_highest_dimension(self.prior, dx = self.psi_dx)
            self.integrated_over_p_and_psi = self.integrate_highest_dimension(self.integrated_over_psi, dx = self.p_dx)
    
            # Normalize prior over domain
            self.normed_prior = self.prior/self.integrated_over_p_and_psi

        except TypeError:
            if self.rht_data is None:
                print("Index {} not found".format(hp_index))
            else:
                print("Unknown TypeError when constructing RHT prior for index {}".format(hp_index))
                
class PriorThetaRHT(BayesianComponent):
    """
    Class for building RHT priors which are defined by theta_RHT and corresponding error
    """
    
    def __init__(self, hp_index, sample_p0, reverse_RHT = False, verbose = False, region = "SC_241", QRHT_cursor = None, URHT_cursor = None, sig_QRHT_cursor = None, sig_URHT_cursor = None):
    
        BayesianComponent.__init__(self, hp_index, verbose = verbose)
        
        # Load Q_RHT, U_RHT, and errors 
        #QRHT_cursor, URHT_cursor, sig_QRHT_cursor, sig_URHT_cursor = get_rht_QU_cursors()
        
        try:
            (self.hp_index, self.QRHT) = QRHT_cursor.execute("SELECT * FROM QRHT WHERE id = ?", (self.hp_index,)).fetchone()
            (self.hp_index, self.URHT) = URHT_cursor.execute("SELECT * FROM URHT WHERE id = ?", (self.hp_index,)).fetchone()
            (self.hp_index, self.QRHTsq) = sig_QRHT_cursor.execute("SELECT * FROM QRHTsq WHERE id = ?", (self.hp_index,)).fetchone()
            (self.hp_index, self.URHTsq) = sig_URHT_cursor.execute("SELECT * FROM URHTsq WHERE id = ?", (self.hp_index,)).fetchone()

            try:
                self.sig_psi, self.sig_P = polarization_tools.sigma_psi_P(self.QRHT, self.URHT, self.QRHTsq, self.URHTsq, degrees = False)
            except ZeroDivisionError:
                print(self.QRHT, self.URHT, self.QRHTsq, self.URHTsq)
        
            # This construction is simple because we can sample everything on [0, pi)
            self.sample_psi0 = np.linspace(0, np.pi, 165)
            self.sample_p0 = sample_p0
        
            # 1D prior will be Gaussian centered on psi_RHT
            self.psimeas = polarization_tools.polarization_angle(self.QRHT, self.URHT, negU = True)
            #gaussian = (1.0/(self.sig_psi*np.sqrt(2*np.pi)))*np.exp(-(self.sample_psi0 - self.psimeas)**2/(2*self.sig_psi**2))
        
            # Instead of gaussian, construct axial von mises distribution
            kappa = 1/self.sig_psi**2
            #vonmises = np.exp(kappa*np.cos(self.sample_psi0 - self.psimeas))/(2*np.pi*special.iv(0, kappa))
            axialvonmises = np.cosh(kappa*np.cos(self.sample_psi0 - self.psimeas))/(np.pi*special.iv(0, kappa))
        
            # Create correct prior geometry
            npsample = len(self.sample_p0)
            #self.prior = np.array([gaussian]*npsample).T
            self.prior = np.array([axialvonmises]*npsample).T
        
            self.psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
            self.p_dx = self.sample_p0[1] - self.sample_p0[0]
        
            if self.psi_dx < 0:
                print("Multiplying psi_dx by -1")
                self.psi_dx *= -1
        
            if verbose is True:
                print("psi dx is {}, p dx is {}".format(self.psi_dx, self.p_dx))
        
            self.integrated_over_psi = self.integrate_highest_dimension(self.prior, dx = self.psi_dx)
            self.integrated_over_p_and_psi = self.integrate_highest_dimension(self.integrated_over_psi, dx = self.p_dx)

            # Normalize prior over domain
            self.normed_prior = self.prior/self.integrated_over_p_and_psi
        
        except TypeError:
            if self.QRHT is None:
                print("Index {} not found".format(hp_index))
            else:
                print("Unknown TypeError when constructing RHT prior for index {}".format(hp_index))
        
               
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
        
        self.psimeas = psimeas
        self.pmeas = pmeas
    
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
    
    def __init__(self, hp_index, sample_p0 = None, adaptivep0 = False, region = "SC_241", useprior = "RHTPrior", rht_cursor = None, QRHT_cursor = None, URHT_cursor = None, sig_QRHT_cursor = None, sig_URHT_cursor = None, gausssmooth_prior = False):
        BayesianComponent.__init__(self, hp_index)  
        
        if sample_p0 is None:
            if adaptivep0 is True:
                self.sample_p0 = self.get_adaptive_p_grid(hp_index)
            else:
                self.sample_p0 = np.linspace(0, 1, 165)
        else:
            self.sample_p0 = sample_p0
        
        # Instantiate posterior components
        if useprior is "RHTPrior":
            prior = Prior(hp_index, self.sample_p0, reverse_RHT = True, region = region, rht_cursor = rht_cursor, gausssmooth = gausssmooth_prior)
        elif useprior is "ThetaRHT":
            prior = PriorThetaRHT(hp_index, self.sample_p0, reverse_RHT = True, region = region, QRHT_cursor = QRHT_cursor, URHT_cursor = URHT_cursor, sig_QRHT_cursor = sig_QRHT_cursor, sig_URHT_cursor = sig_URHT_cursor)
            
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
        self.psimeas = likelihood.psimeas
        self.pmeas = likelihood.pmeas
        
        self.normed_prior = prior.normed_prior#/np.max(prior.normed_prior)
        self.planck_likelihood = likelihood.likelihood
        
        #self.posterior = np.einsum('ij,jk->ik', self.planck_likelihood, self.normed_prior)
        self.posterior = self.planck_likelihood*self.normed_prior
        
        psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
        p_dx = self.sample_p0[1] - self.sample_p0[0]
        
        self.posterior_integrated_over_psi = self.integrate_highest_dimension(self.posterior, dx = psi_dx)
        self.posterior_integrated_over_p_and_psi = self.integrate_highest_dimension(self.posterior_integrated_over_psi, dx = p_dx)
        
        self.normed_posterior = self.posterior/self.posterior_integrated_over_p_and_psi
        
        self.prior_obj = prior
        
class PlanckPosterior(BayesianComponent):
    """
    Class for building a posterior that is only a Planck-based likelihood
    """
    def __init__(self, hp_index, planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all, adaptivep0 = True):
        BayesianComponent.__init__(self, hp_index)      
    
        # Planck-based likelihood
        if adaptivep0 is True:
            self.sample_p0 = self.get_adaptive_p_grid(hp_index)
        else:
            self.sample_p0 = p0_all
        self.sample_psi0 = psi0_all
        
        likelihood = Likelihood(hp_index, planck_tqu_cursor, planck_cov_cursor, self.sample_p0, self.sample_psi0)
        self.posterior = likelihood.likelihood
    
        self.naive_psi = likelihood.naive_psi
        self.psimeas = likelihood.psimeas
        self.pmeas = likelihood.pmeas
    
        psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
        p_dx = self.sample_p0[1] - self.sample_p0[0]
    
        self.posterior_integrated_over_psi = self.integrate_highest_dimension(self.posterior, dx = psi_dx)
        self.posterior_integrated_over_p_and_psi = self.integrate_highest_dimension(self.posterior_integrated_over_psi, dx = p_dx)
    
        self.normed_posterior = self.posterior/self.posterior_integrated_over_p_and_psi
        
class DummyPosterior(BayesianComponent):
      """
      Class for testing posterior estimation methods. 
      """
      
      def __init__(self):
        BayesianComponent.__init__(self, 0)  
        
        self.sample_p0 = np.linspace(0, 1, 165)
        self.sample_psi0 = np.linspace(0, np.pi, 165)
    
        self.psi_dx = self.sample_psi0[1] - self.sample_psi0[0]
        self.p_dx = self.sample_p0[1] - self.sample_p0[0]
        
        if self.psi_dx < 0:
            print("Multiplying psi_dx by -1")
            self.psi_dx *= -1
        
        print("psi dx is {}, p dx is {}".format(self.psi_dx, self.p_dx))
        
        psi_y = self.sample_psi0[:, np.newaxis]
        p_x = self.sample_p0
        
        self.psimeas = np.pi/3
        self.pmeas = 0.2
        
        self.fwhm = 0.3
        
        gaussian = np.exp(-4*np.log(2) * ((p_x-self.pmeas)**2 + (psi_y-self.psimeas)**2) / self.fwhm**2)
        
        self.planck_likelihood = gaussian
        
        self.integrated_over_psi = self.integrate_highest_dimension(self.planck_likelihood, dx = self.psi_dx)
        self.integrated_over_p_and_psi = self.integrate_highest_dimension(self.integrated_over_psi, dx = self.p_dx)
        
        self.normed_posterior = self.planck_likelihood/self.integrated_over_p_and_psi
        
        self.normed_prior = np.ones(self.normed_posterior.shape, np.float_)
      

def latex_formatter(x, pos):
    return "${0:.1f}$".format(x)

def plot_bayesian_component_from_posterior(posterior_obj, component = "posterior", ax = None, cmap = "cubehelix"):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    #extent = [posterior_obj.sample_p0[0], posterior_obj.sample_p0[-1], posterior_obj.sample_psi0[0], posterior_obj.sample_psi0[-1]]
    #aspect = (posterior_obj.sample_p0[1] - posterior_obj.sample_p0[0])/(posterior_obj.sample_psi0[1] - posterior_obj.sample_psi0[0])
    
    ax.set_aspect("auto")
    
    if component == "posterior":
        plotarr = posterior_obj.normed_posterior
        title = r"$\mathrm{Posterior}$"
    if component == "likelihood":
        plotarr = posterior_obj.planck_likelihood
        title = r"$\mathrm{Planck}$ $\mathrm{Likelihood}$"
    if component == "prior":
        plotarr = posterior_obj.normed_prior
        title = r"$\mathrm{RHT}$ $\mathrm{Prior}$"
    
    # Plotting grid needs to be mod pi unless it's exactly 0 to pi
    if posterior_obj.sample_psi0[-1] == np.pi: 
        im = ax.pcolor(posterior_obj.sample_p0, posterior_obj.sample_psi0, plotarr, cmap = cmap)
    else:
        im = ax.pcolor(posterior_obj.sample_p0, np.mod(posterior_obj.sample_psi0, np.pi), plotarr, cmap = cmap)
    
    ax.set_title(title, size = 15)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="15%", pad=0.05)#, aspect = 100./15)
    cbar = plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(latex_formatter))
    
def plot_all_bayesian_components_from_posterior(posterior_obj, cmap = "cubehelix"):
    
    fig = plt.figure(figsize = (14, 4), facecolor = "white")
    gs = gridspec.GridSpec(1, 3)
    ax1 = plt.subplot(gs[0])#fig.add_subplot(131)
    ax2 = plt.subplot(gs[1])#fig.add_subplot(132)
    ax3 = plt.subplot(gs[2])#fig.add_subplot(133)
    gs.update(left=0.05, right=0.95, wspace=0.3, hspace = 0.3, bottom = 0.15)
    
    plot_bayesian_component_from_posterior(posterior_obj, component = "likelihood", ax = ax1, cmap = cmap)
    plot_bayesian_component_from_posterior(posterior_obj, component = "prior", ax = ax2, cmap = cmap)
    plot_bayesian_component_from_posterior(posterior_obj, component = "posterior", ax = ax3, cmap = cmap)
    
    #plt.subplots_adjust(wspace = 0.2)
    
    pMB, psiMB = mean_bayesian_posterior(posterior_obj, center = "naive")
    ax3.plot(pMB, np.mod(psiMB, np.pi), '+', ms = 10, mew = 2, color = "gray")
    
    pMB, psiMB = mean_bayesian_posterior(posterior_obj, center = "MAP")
    ax3.plot(pMB, np.mod(psiMB, np.pi), '+', ms = 10, mew = 2, color = "teal")
    
    p_map, psi_map = maximum_a_posteriori(posterior_obj)
    ax3.plot(p_map, np.mod(psi_map, np.pi), '+', ms = 10, mew = 2, color = "red")
    
    #pnaive, psinaive = naive_planck_measurements(posterior_obj.hp_index)
    pnaive = posterior_obj.pmeas
    psinaive = posterior_obj.psimeas
    ax1.plot(pnaive, psinaive, '+', ms = 10, mew = 2, color = "gray")
    
    axs = [ax1, ax2, ax3]
    for ax in axs:
        if np.mod(posterior_obj.sample_psi0[-1], np.pi) == 0:
            ax.set_ylim(np.mod(posterior_obj.sample_psi0[0], np.pi), np.pi)
        else:
            ax.set_ylim(np.mod(posterior_obj.sample_psi0[0], np.pi), np.mod(posterior_obj.sample_psi0[-1], np.pi))
        ax.set_ylabel(r"$\psi_0$", size = 15)
        ax.set_xlabel(r"$p_0$", size = 15)
        
def naive_planck_measurements(hp_index, verbose=False):
    
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

    if verbose is True:
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
    
def center_posterior_naive_psi(posterior_obj, sample_psi0, posterior, verbose = False):

    try:
        pnaive = posterior_obj.pmeas
        psinaive = posterior_obj.psimeas
    except AttributeError:
        print("Obtaining naive measurements from posterior object")
        pnaive, psinaive = naive_planck_measurements(posterior_obj.hp_index)
    
    # Find index of value closest to psinaive - pi/2
    psinaive_indx = np.abs(sample_psi0 - (psinaive - np.pi/2)).argmin()
    
    # catch pathological case
    if (psinaive - np.pi/2) <= 1E-10:
        if verbose is True:
            print("psinaive is EXACTLY pi/2. Doing nothing.")
    else:
    
        if verbose is True:
            print("difference between psinaive - pi/2 and closest values is {} - {} = {}".format(psinaive - np.pi/2, sample_psi0[psinaive_indx], np.abs((psinaive - np.pi/2) - sample_psi0[psinaive_indx])))
        if np.abs((psinaive - np.pi/2) - sample_psi0[psinaive_indx]) > (sample_psi0[1] - sample_psi0[0]):
            if verbose is True:
                print("Subtracting pi from all")
            sample_psi0 -= np.pi
            psinaive_indx = np.abs(sample_psi0 - (psinaive - np.pi/2)).argmin()
            if verbose is True:
                print("Redefining psinaive_indx")
                print("difference between psinaive - pi/2 and closest values is {} - {} = {}".format(psinaive - np.pi/2, sample_psi0[psinaive_indx], np.abs((psinaive - np.pi/2) - sample_psi0[psinaive_indx])))
    
    rolled_posterior = np.roll(posterior, - psinaive_indx, axis = 0)
    
    rolled_sample_psi0 = np.roll(sample_psi0, - psinaive_indx)
    rolled_sample_psi0[rolled_sample_psi0 < psinaive - np.pi/2] += np.pi
    rolled_sample_psi0[rolled_sample_psi0 > psinaive + np.pi/2] -= np.pi
    
    return rolled_sample_psi0, rolled_posterior 
    
def center_posterior_psi_MAP(posterior_obj, sample_psi0, posterior, verbose = False):

    pMAP, psiMAP = maximum_a_posteriori(posterior_obj)

    # Find index of value closest to psiMAP - pi/2
    psiMAP_indx = np.abs(sample_psi0 - (psiMAP - np.pi/2)).argmin()
    
    if verbose is True:
        print("difference between psiMAP - pi/2 and closest values is {} - {} = {}".format(psiMAP - np.pi/2, sample_psi0[psiMAP_indx], np.abs((psiMAP - np.pi/2) - sample_psi0[psiMAP_indx])))
    
    if np.abs((psiMAP - np.pi/2) - sample_psi0[psiMAP_indx]) > (sample_psi0[1] - sample_psi0[0]):
        if verbose is True:
            print("Subtracting pi from all")
        sample_psi0 -= np.pi
        psiMAP_indx = np.abs(sample_psi0 - (psiMAP - np.pi/2)).argmin()
        
        if verbose is True:
            print("Redefining psiMAP_indx")
            print("difference between psiMAP - pi/2 and closest values is {} - {} = {}".format(psiMAP - np.pi/2, sample_psi0[psiMAP_indx], np.abs((psiMAP - np.pi/2) - sample_psi0[psiMAP_indx])))
    
    rolled_posterior = np.roll(posterior, - psiMAP_indx, axis = 0)
    
    rolled_sample_psi0 = np.roll(sample_psi0, - psiMAP_indx)
    rolled_sample_psi0[rolled_sample_psi0 < psiMAP - np.pi/2] += np.pi
    rolled_sample_psi0[rolled_sample_psi0 > psiMAP + np.pi/2] -= np.pi
    
    return rolled_sample_psi0, rolled_posterior
    
def center_posterior_psi_given_old(sample_psi0, posterior, given_psi, verbose = False):
    """
    Center posterior on given psi
    """

    # Find index of value closest to given_psi - pi/2
    given_psi_indx = np.abs(sample_psi0 - (given_psi - np.pi/2)).argmin()
    
    if verbose is True:
        print("difference between given_psi - pi/2 and closest values is {} - {} = {}".format(given_psi - np.pi/2, sample_psi0[given_psi_indx], np.abs((given_psi - np.pi/2) - sample_psi0[given_psi_indx])))
    if np.abs((given_psi - np.pi/2) - sample_psi0[given_psi_indx]) > (sample_psi0[1] - sample_psi0[0]):
        if verbose is True:
            print("Subtracting pi from all")
        sample_psi0 -= np.pi
        given_psi_indx = np.abs(sample_psi0 - (given_psi - np.pi/2)).argmin()
        if verbose is True:
            print("Redefining given_psi_indx")
            print("difference between given_psi - pi/2 and closest values is {} - {} = {}".format(given_psi - np.pi/2, sample_psi0[given_psi_indx], np.abs((given_psi - np.pi/2) - sample_psi0[given_psi_indx])))
    
    rolled_posterior = np.roll(posterior, - given_psi_indx, axis = 0)
    
    rolled_sample_psi0 = np.roll(sample_psi0, - given_psi_indx)
    rolled_sample_psi0[rolled_sample_psi0 < given_psi - np.pi/2] += np.pi
    rolled_sample_psi0[rolled_sample_psi0 > given_psi + np.pi/2] -= np.pi
    
    return rolled_sample_psi0, rolled_posterior
    
def center_posterior_psi_given(sample_psi0, posterior, given_psi, verbose = False):
    """
    Center posterior on given psi
    """
    
    psi0new = np.linspace(given_psi - np.pi/2, given_psi + np.pi/2, len(sample_psi0), endpoint=True)
    
    centered_posterior = np.zeros(posterior.shape)
    for i, col in enumerate(posterior.T):
        centered_posterior[:, i] = np.interp(psi0new, sample_psi0, col, period=np.pi)
        
    return psi0new, centered_posterior 
    
def circular_integration(xpoints, ypoints, endpoint_included=False, axis=0):
    """
    Integrate on wrapped domain. endpoint_included should be false if array has linspace endpoint=False
    """
    if endpoint_included is False:
        integrateme = np.zeros(len(ypoints)+1)
        integrateme[:-1] = ypoints
        integrateme[-1] = ypoints[0] # wrap domain
        
        integratex = np.zeros(len(xpoints) + 1)
        integratex[:-1] = xpoints
        integratex[-1] = xpoints[-1] + (xpoints[1]-xpoints[0])
    elif endpoint_included is True:
        integrateme = ypoints
        integratex = xpoints
    
    intdata = np.trapz(integrateme, integratex, axis=axis)
    
    return intdata
    
def maximum_a_posteriori(posterior_obj, verbose = False):
    """
    MAP estimator
    """
    
    #psi_map_indx = scipy.stats.mode(np.argmax(posterior_obj.normed_posterior, axis=0))[0][0]
    #p_map_indx = scipy.stats.mode(np.argmax(posterior_obj.normed_posterior, axis=1))[0][0]
    
    psi_map_indx, p_map_indx = np.where(posterior_obj.normed_posterior == np.nanmax(posterior_obj.normed_posterior))
    psi_map_indx = psi_map_indx[0]
    p_map_indx = p_map_indx[0]
    
    psi_map = posterior_obj.sample_psi0[psi_map_indx]
    p_map = posterior_obj.sample_p0[p_map_indx]
    
    if verbose is True:
        print("pMAP is {}".format(p_map))
        print("psiMAP is {}".format(psi_map))
    
    return p_map, psi_map
    
def mean_bayesian_posterior(posterior_obj, center = "naive", verbose = False, tol=1E-5):
    """
    Integrated first order moments of the posterior PDF
    """
    
    posterior = copy.copy(posterior_obj.normed_posterior)
    
    sample_p0 = posterior_obj.sample_p0
    sample_psi0 = posterior_obj.sample_psi0
    
    # Sampling widths
    pdx = sample_p0[1] - sample_p0[0]
    psidx = sample_psi0[1] - sample_psi0[0]
    
    # pMB integrand is p0*B2D. This can happen once only, before centering.
    pMB_integrand = posterior*sample_p0
    pMB_integrated_over_psi0 = posterior_obj.integrate_highest_dimension(pMB_integrand, dx = psidx)
    pMB = posterior_obj.integrate_highest_dimension(pMB_integrated_over_psi0, dx = pdx)
    
    if verbose is True:
        print("Sampling pdx is {}, psidx is {}".format(pdx, psidx))
    
    # Test that normed posterior is normed
    if verbose is True:
        norm_posterior_test = test_normalization(posterior_obj, pdx, psidx)
    
    # Center on the naive psi
    if center == "naive":
        if verbose is True:
            print("Centering initial integral on naive psi")
        #rolled_sample_psi0, rolled_posterior = center_posterior_naive_psi(posterior_obj, sample_psi0, posterior, verbose = verbose)
        #pnaive, psinaive = naive_planck_measurements(posterior_obj.hp_index)
        psinaive = posterior.psimeas
        pnaive = posterior.pmeas
        psi0new, centered_posterior = center_posterior_psi_given(sample_psi0, posterior, psinaive, verbose = verbose)
        
    elif center == "MAP":
        print("WARNING: MAP center may not be correctly implemented")
        if verbose is True:
            print("Centering initial integral on psi_MAP")
        rolled_sample_psi0, rolled_posterior = center_posterior_psi_MAP(posterior_obj, sample_psi0, posterior, verbose = verbose)
    
    #posterior = rolled_posterior
    #sample_psi0 = rolled_sample_psi0
    
    # Integrate over p
    #pMB1 = np.trapz(posterior, dx = psidx, axis = 0)
    #pMB1 = np.trapz(centered_posterior, psi0new, axis=0)
    
    # Integrate over psi
    #pMB = np.trapz(pMB1*sample_p0, dx = pdx)
    
    # Integrate over p
    #psiMB1 = np.trapz(posterior, dx = pdx, axis = 1)
    #psiMB1 = np.trapz(centered_posterior, dx = pdx, axis = 1)
    
    # Integrate over psi
    #psiMB = np.trapz(psiMB1*sample_psi0, dx = psidx)
    #psiMB = np.trapz(psiMB1*psi0new, psi0new)

    #test
    if psidx != psi0new[1] - psi0new[0]:
        print("Caution: old psidx = {}, new psidx = {}".format(psidx, psi0new[1] - psi0new[0]))
    
    # psiMB integrand is psi0*B2D.
    psiMB_integrand = centered_posterior*psi0new[:, np.newaxis]
    psiMB_integrated_over_psi0 = self.integrate_highest_dimension(psiMB_integrand, dx=psidx)
    psiMB = self.integrate_highest_dimension(psiMB_integrated_over_psi0, dx=pdx)
    
    if verbose is True:
        print("initial pMB is {}".format(pMB))
        print("initial psiMB is {}".format(psiMB))
    
    # Set parameters for convergence
    psi_last = psiMB - np.pi/2
    i = 0
    itertol = 10000
    if verbose is True:
        print("Using tolerance of {}".format(tol))
        
    while (np.abs(angle_residual(np.mod(psi_last, np.pi), np.mod(psiMB, np.pi), degrees = False)) > tol) and (i < itertol):
        if verbose is True:
            print("Convergence at {}".format(np.abs(np.mod(psi_last, np.pi) - np.mod(psiMB, np.pi))))
            print("i = {}".format(i))
        psi_last = copy.copy(psiMB)
        centered_posterior_last = copy.copy(centered_posterior)
    
        #rolled_sample_psi0, rolled_posterior = center_posterior_psi_given(rolled_sample_psi0, rolled_posterior, np.mod(psiMB, np.pi), verbose = verbose)
        psi0new, centered_posterior = center_posterior_psi_given(psi0new, centered_posterior_last, psi_last, verbose = verbose)

        psiMB_integrand = centered_posterior*psi0new[:, np.newaxis]
        psiMB_integrated_over_psi0 = posterior_obj.integrate_highest_dimension(psiMB_integrand, dx=psidx)
        psiMB = posterior_obj.integrate_highest_dimension(psiMB_integrated_over_psi0, dx=pdx)
        
        if verbose is True:
            print("Iterating. New pMB is {}".format(pMB))
            print("Iterating. New psiMB is {}".format(psiMB))
        i += 1
        
        if i >= itertol-1:
            print("CAUTION: i is now {}. Index {} may not converge".format(i, posterior_obj.hp_index))
            print("psi initial = {}, psi last = {}, psiMB = {}".format(psinaive, np.mod(psi_last, np.pi), np.mod(psiMB, np.pi)))
            print("greater than tol: {}".format(np.abs(angle_residual(np.mod(psi_last, np.pi), np.mod(psiMB, np.pi), degrees = False)))) 
    
    #print("difference between original and final psi is {}".format(angle_residual(psiMB, psinaive, degrees=False)))
    #print("difference between original and final p is {}".format(pMB - pnaive))
    if i >= itertol-1:
        pMB = copy.copy(pnaive)
        psiMB = copy.copy(psinaive)
        print("Iteration tolerance reached. setting naive values")
        
    return pMB, psiMB#, pMB1, psiMB1, sample_psi0, sample_p0

def test_normalization(posterior_obj, pdx, psidx):
    norm_posterior_test = posterior_obj.integrate_highest_dimension(posterior_obj.normed_posterior, dx = psidx)
    norm_posterior_test = posterior_obj.integrate_highest_dimension(norm_posterior_test, dx = pdx)
    
    print("Normalized posterior is {}".format(norm_posterior_test))
    
    return norm_posterior_test
    
def get_all_rht_ids(rht_cursor, tablename):
    all_ids = rht_cursor.execute("SELECT id from "+tablename).fetchall()
    
    return all_ids
    
def get_rht_cursor(region = "SC_241", velrangestring = "-10_10", local=False):
    if region is "SC_241":
        rht_db = sqlite3.connect("allweights_db.sqlite")
        tablename = "RHT_weights"
    elif region is "allsky":
        if local is True:
            root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/"
        else:
            root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
        tablename = "RHT_weights_allsky"
        if velrangestring == "-10_10":
            rht_db = sqlite3.connect(root + "allsky_RHTweights_db.sqlite")
        elif velrangestring == "-4_3":
            print("Loading database with velrangestring -4_3")
            rht_db = sqlite3.connect(root + "allsky_RHTweights_-4_3_db.sqlite")
    
    rht_cursor = rht_db.cursor()
    
    return rht_cursor, tablename
    
def get_rht_QU_cursors(local = False):
    
    if local is True:
        root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/QUmaps/"
        print(root)
    else:
        root = "/disks/jansky/a/users/goldston/susan/Planck/code/"
    
    print("fn is ", root + "QRHT_ch1004_1043_db.sqlite")
    QRHT_db = sqlite3.connect(root + "QRHT_ch1004_1043_db.sqlite")
    URHT_db = sqlite3.connect(root + "URHT_ch1004_1043_db.sqlite")
    sig_QRHT_db = sqlite3.connect(root + "QRHTsq_ch1004_1043_db.sqlite")
    sig_URHT_db = sqlite3.connect(root + "URHTsq_ch1004_1043_db.sqlite")
    
    QRHT_cursor = QRHT_db.cursor()
    URHT_cursor = URHT_db.cursor()
    sig_QRHT_cursor = sig_QRHT_db.cursor()
    sig_URHT_cursor = sig_URHT_db.cursor()

    return QRHT_cursor, URHT_cursor, sig_QRHT_cursor, sig_URHT_cursor

def sample_all_rht_points(all_ids, adaptivep0 = True, rht_cursor = None, region = "SC_241", useprior = "RHTPrior", gausssmooth_prior = False, tol=1E-5):
    
    all_pMB = np.zeros(len(all_ids))
    all_psiMB = np.zeros(len(all_ids))
    
    # Get ids of all pixels that contain RHT data
    if rht_cursor is None:
        print("Loading default rht_cursor by region because it was not provided")
        rht_cursor, tablename = get_rht_cursor(region = region)
        
    update_progress(0.0)
    for i, _id in enumerate(all_ids):
        posterior_obj = Posterior(_id[0], adaptivep0 = adaptivep0, region = region, useprior = useprior, rht_cursor = rht_cursor, gausssmooth_prior = gausssmooth_prior)
        all_pMB[i], all_psiMB[i] = mean_bayesian_posterior(posterior_obj, center = "naive", verbose = False, tol=tol)
        update_progress((i+1.0)/len(all_ids), message='Sampling: ', final_message='Finished Sampling: ')
    
    return all_pMB, all_psiMB
    
def sample_all_planck_points(all_ids, adaptivep0 = True, planck_tqu_cursor = None, planck_cov_cursor = None, region = "SC_241", verbose = False, tol=1E-5):
    """
    Sample the Planck likelihood rather than a posterior constructed from a likelihood and prior
    """

    all_pMB = np.zeros(len(all_ids))
    all_psiMB = np.zeros(len(all_ids))

    if planck_tqu_cursor is None:
        print("Loading default planck_tqu_cursor because it was not provided")
        planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
        planck_tqu_cursor = planck_tqu_db.cursor()
    
    if planck_cov_cursor is None:
        print("Loading default planck_cov_cursor because it was not provided")
        planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
        planck_cov_cursor = planck_cov_db.cursor()

    # Get p0 and psi0 sampling grids
    p0_all = np.linspace(0, 1, 165)
    psi0_all = np.linspace(0, np.pi, 165, endpoint=False) # don't count both 0 and pi

    update_progress(0.0)
    for i, _id in enumerate(all_ids):
        #if _id[0] in [3400757, 793551, 2447655]:
        posterior_obj = PlanckPosterior(_id[0], planck_tqu_cursor, planck_cov_cursor, p0_all, psi0_all, adaptivep0 = adaptivep0)
        #print("for id {}, p0 grid is {}".format(_id, posterior_obj.sample_p0))
        #print("for id {}, pmeas is {}, psimeas is {}, psi naive is {}".format(_id, posterior_obj.pmeas, posterior_obj.psimeas, posterior_obj.naive_psi))
        #testing
        all_pMB[i], all_psiMB[i] = mean_bayesian_posterior(posterior_obj, center = "naive", verbose = verbose, tol=tol)
        if verbose is True:
            print("for id {}, num {}, I get pMB {} and psiMB {}".format(_id, i, all_pMB[i], all_psiMB[i]))
        
        update_progress((i+1.0)/len(all_ids), message='Sampling: ', final_message='Finished Sampling: ')
    
    return all_pMB, all_psiMB
    
def sample_all_rht_points_ThetaRHTPrior(all_ids, adaptivep0 = True, region = "SC_241", useprior = "RHTPrior", local = False, tol=1E-5):
    
    all_pMB = np.zeros(len(all_ids))
    all_psiMB = np.zeros(len(all_ids))
    
    # Get ids of all pixels that contain RHT data
    QRHT_cursor, URHT_cursor, sig_QRHT_cursor, sig_URHT_cursor = get_rht_QU_cursors(local = local)
    
    update_progress(0.0)
    for i, _id in enumerate(all_ids):
        posterior_obj = Posterior(_id[0], adaptivep0 = adaptivep0, region = region, useprior = useprior, QRHT_cursor = QRHT_cursor, URHT_cursor = URHT_cursor, sig_QRHT_cursor = sig_QRHT_cursor, sig_URHT_cursor = sig_URHT_cursor)
        all_pMB[i], all_psiMB[i] = mean_bayesian_posterior(posterior_obj, center = "naive", verbose = False, tol=tol)
        update_progress((i+1.0)/len(all_ids), message='Sampling: ', final_message='Finished Sampling: ')
        
    return all_pMB, all_psiMB
    
def fully_sample_sky(region = "allsky", limitregion = False, adaptivep0 = True, useprior = "RHTPrior", velrangestring = "-10_10", gausssmooth_prior = False, tol=1E-5):
    """
    Sample psi_MB and p_MB from whole GALFA-HI sky
    """
    
    print("Fully sampling sky with options: region = {}, limitregion = {}, useprior = {}, velrangestring = {}, gausssmooth_prior = {}".format(region, limitregion, useprior, velrangestring, gausssmooth_prior))

    # Get ids of all pixels that contain RHT data
    rht_cursor, tablename = get_rht_cursor(region = region, velrangestring = velrangestring)
    all_ids = get_all_rht_ids(rht_cursor, tablename)
    
    if limitregion is True:
        print("Loading all allsky data points that are in the SC_241 region")
        # Get all ids that are in both allsky data and SC_241
        all_ids_SC = pickle.load(open("SC_241_healpix_ids.p", "rb"))
        all_ids = list(set(all_ids).intersection(all_ids_SC))
    
    print("beginning creation of all posteriors")
    
    # Create and sample posteriors for all pixels
    all_pMB, all_psiMB = sample_all_rht_points(all_ids, adaptivep0 = adaptivep0, rht_cursor = rht_cursor, region = region, useprior = useprior, gausssmooth_prior = gausssmooth_prior, tol=tol)
    
    # Place into healpix map
    hp_psiMB = make_hp_map(all_psiMB, all_ids, Nside = 2048, nest = True)
    hp_pMB = make_hp_map(all_pMB, all_ids, Nside = 2048, nest = True)
    
    out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
    if limitregion is False:
        psiMB_out_fn = "psiMB_allsky_"+velrangestring+"_smoothprior_"+str(gausssmooth_prior)+"_adaptivep0_"+str(adaptivep0)+".fits"
        pMB_out_fn = "pMB_allsky_"+velrangestring+"_smoothprior_"+str(gausssmooth_prior)+"_adaptivep0_"+str(adaptivep0)+".fits"
    elif limitregion is True:
        psiMB_out_fn = "psiMB_DR2_SC_241_"+velrangestring+"_smoothprior_"+str(gausssmooth_prior)+"_adaptivep0_"+str(adaptivep0)+"_tol_{}.fits".format(tol)
        pMB_out_fn = "pMB_DR2_SC_241_"+velrangestring+"_smoothprior_"+str(gausssmooth_prior)+"_adaptivep0_"+str(adaptivep0)+"_tol_{}.fits".format(tol)
    hp.fitsfunc.write_map(out_root + psiMB_out_fn, hp_psiMB, coord = "C", nest = True) 
    hp.fitsfunc.write_map(out_root + pMB_out_fn, hp_pMB, coord = "C", nest = True) 
    
def fully_sample_planck_sky(region = "allsky", adaptivep0 = True, limitregion = False, local = False, verbose = False, tol=1E-5):
    """
    Sample Planck 353 GHz psi_MB and p_MB from whole GALFA-HI sky
    """
    
    # Get ids of all pixels that contain RHT data
    rht_cursor, tablename = get_rht_cursor(region = region)
    all_ids = get_all_rht_ids(rht_cursor, tablename)
    
    planck_tqu_db = sqlite3.connect("planck_TQU_gal_2048_db.sqlite")
    planck_tqu_cursor = planck_tqu_db.cursor()
    planck_cov_db = sqlite3.connect("planck_cov_gal_2048_db.sqlite")
    planck_cov_cursor = planck_cov_db.cursor()
    
    if limitregion is True:
        print("Loading all allsky data points that are in the SC_241 region")
        # Get all ids that are in both allsky data and SC_241
        all_ids_SC = pickle.load(open("SC_241_healpix_ids.p", "rb"))
        all_ids = list(set(all_ids).intersection(all_ids_SC))
    
    print("beginning creation of all likelihoods")
    all_pMB, all_psiMB = sample_all_planck_points(all_ids, adaptivep0 = adaptivep0, planck_tqu_cursor = planck_tqu_cursor, planck_cov_cursor = planck_cov_cursor, region = "SC_241", verbose = verbose, tol=tol)
    
    # Place into healpix map
    hp_psiMB = make_hp_map(all_psiMB, all_ids, Nside = 2048, nest = True)
    hp_pMB = make_hp_map(all_pMB, all_ids, Nside = 2048, nest = True)
    
    if local is True:
        out_root = ""
    else:
        out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
        
    if limitregion is False:
        psiMB_out_fn = "psiMB_allsky_353GHz_adaptivep0_"+str(adaptivep0)+".fits"
        pMB_out_fn = "pMB_allsky_353GHz_adaptivep0_"+str(adaptivep0)+".fits"
    elif limitregion is True:
        psiMB_out_fn = "psiMB_DR2_SC_241_353GHz_adaptivep0_"+str(adaptivep0)+"_tol_{}.fits".format(tol)
        pMB_out_fn = "pMB_DR2_SC_241_353GHz_adaptivep0_"+str(adaptivep0)+"_tol_{}.fits".format(tol)
    hp.fitsfunc.write_map(out_root + psiMB_out_fn, hp_psiMB, coord = "C", nest = True) 
    hp.fitsfunc.write_map(out_root + pMB_out_fn, hp_pMB, coord = "C", nest = True) 

    
def gauss_sample_sky(region = "allsky", useprior = "ThetaRHT"):
    
    # Get ids of all pixels that contain RHT data
    QRHT_cursor, URHT_cursor, sig_QRHT_cursor, sig_URHT_cursor = get_rht_QU_cursors()
    all_ids_QRHT = get_all_rht_ids(QRHT_cursor, "QRHT")
    all_ids_URHT = get_all_rht_ids(URHT_cursor, "URHT")
    all_ids_QRHTsq = get_all_rht_ids(sig_QRHT_cursor, "QRHTsq")
    all_ids_URHTsq = get_all_rht_ids(sig_URHT_cursor, "URHTsq")
    all_ids = list(set(all_ids_QRHT).intersection(all_ids_URHT).intersection(all_ids_QRHTsq).intersection(all_ids_URHTsq))
    
    # Create and sample posteriors for all pixels
    all_pMB, all_psiMB = sample_all_rht_points_ThetaRHTPrior(all_ids, region = region, useprior = useprior)
    
    # Place into healpix map
    hp_psiMB = make_hp_map(all_psiMB, all_ids, Nside = 2048, nest = True)
    hp_pMB = make_hp_map(all_pMB, all_ids, Nside = 2048, nest = True)
    
    out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
    hp.fitsfunc.write_map(out_root + "psiMB_SC_241_thetaRHT_test0.fits", hp_psiMB, coord = "C", nest = True) 
    hp.fitsfunc.write_map(out_root + "pMB_SC_241_thetaRHT_test0.fits", hp_pMB, coord = "C", nest = True) 
    
def gauss_sample_region(region = "SC_241", useprior = "ThetaRHT", local = True):
    
    # Get ids of all pixels that contain RHT data
    QRHT_cursor, URHT_cursor, sig_QRHT_cursor, sig_URHT_cursor = get_rht_QU_cursors(local = local)
    all_ids_QRHT = get_all_rht_ids(QRHT_cursor, "QRHT")
    all_ids_URHT = get_all_rht_ids(URHT_cursor, "URHT")
    all_ids_QRHTsq = get_all_rht_ids(sig_QRHT_cursor, "QRHTsq")
    all_ids_URHTsq = get_all_rht_ids(sig_URHT_cursor, "URHTsq")
    all_ids = list(set(all_ids_QRHT).intersection(all_ids_URHT).intersection(all_ids_QRHTsq).intersection(all_ids_URHTsq))
    
    # Get ids of all pixels that are in SC_241
    #rht_cursor_SC, tablename_SC = get_rht_cursor(region = "SC_241")
    #all_ids_SC = get_all_rht_ids(rht_cursor_SC, tablename_SC)
    all_ids_SC = pickle.load(open("SC_241_healpix_ids.p", "rb"))
    
    # Get all ids that are in both allsky data and SC_241
    all_ids_set = list(set(all_ids).intersection(all_ids_SC))
    
    # Create and sample posteriors for all pixels
    all_pMB, all_psiMB = sample_all_rht_points_ThetaRHTPrior(all_ids_set, region = region, useprior = useprior, local = local)
    
    # Place into healpix map
    hp_psiMB = make_hp_map(all_psiMB, all_ids_set, Nside = 2048, nest = True)
    hp_pMB = make_hp_map(all_pMB, all_ids_set, Nside = 2048, nest = True)
    
    if local is True:
        out_root = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/"
    else:
        out_root = "/disks/jansky/a/users/goldston/susan/Wide_maps/"
    hp.fitsfunc.write_map(out_root + "psiMB_SC_241_thetaRHT_test0.fits", hp_psiMB, coord = "C", nest = True) 
    hp.fitsfunc.write_map(out_root + "pMB_SC_241_thetaRHT_test0.fits", hp_pMB, coord = "C", nest = True) 

    
def make_hp_map(data, hp_indices, Nside = 2048, nest = True):
    """
    Places data into array of healpix pixels by healpix index.
    """
    
    Npix = 12*Nside**2
    map_data = np.zeros(Npix, np.float_)
    map_data[hp_indices] = data
    
    return map_data

def sampled_data_to_hp(psiMB, pMB, hp_indices, nest = True):
    """
    Write data to healpix map. Wraps make_hp_map
    """
    
    hp_psiMB = make_hp_map(psiMB, hp_indices, Nside = 2048, nest = nest)
    hp_pMB = make_hp_map(pMB, hp_indices, Nside = 2048, nest = nest)
    
    out_root = "/Users/susanclark/BetterForegrounds/data/"
    hp.fitsfunc.write_map(out_root + "psiMB_test0.fits", hp_psiMB, coord = "C", nest = nest) 
    hp.fitsfunc.write_map(out_root + "pMB_test0.fits", hp_pMB, coord = "C", nest = nest) 
    
def angle_residual(ang1, ang2, degrees = True):
    if degrees is True:
        ang1 = np.radians(ang1)
        ang2 = np.radians(ang2)

    dang_num = (np.sin(2*ang1)*np.cos(2*ang2) - np.cos(2*ang1)*np.sin(2*ang2))
    dang_denom = (np.cos(2*ang1)*np.cos(2*ang2) + np.sin(2*ang1)*np.sin(2*ang2))
    dang = 0.5*np.arctan2(dang_num, dang_denom)
    
    if degrees is True:
        dang = np.degrees(dang)
    
    return dang

def update_progress(progress, message='Progress:', final_message='Finished:'):
    # Create progress meter that looks like: 
    # message + ' ' + '[' + '#'*p + ' '*(length-p) + ']' + time_message

    if not 0.0 <= progress <= 1.0:
        # Fast fail for values outside the allowed range
        raise ValueError('Progress value outside allowed value in update_progress') 

    # Slow Global Implementation
    global start_time
    global stop_time 

    # First call
    if 0.0 == progress:
        start_time = time.time()
        stop_time = None
        return

    # Second call
    elif stop_time is None:
        stop_time = start_time + (time.time() - start_time)/progress

    # Randomly callable re-calibration
    elif np.random.rand() > 0.98: 
        stop_time = start_time + (time.time() - start_time)/progress

    # Normal Call with Progress
    sec_remaining = int(stop_time - time.time())
    if sec_remaining >= 60:
        time_message = ' < ' + str(sec_remaining//60  +1) + 'min'
    else:
        time_message = ' < ' + str(sec_remaining +1) + 'sec'

    TEXTWIDTH = 70
    length = int(0.55 * TEXTWIDTH)
    messlen = TEXTWIDTH-(length+3)-len(time_message)
    message = string.ljust(message, messlen)[:messlen]

    p = int(length*progress/1.0) 
    sys.stdout.write('\r{2} [{0}{1}]{3}'.format('#'*p, ' '*(length-p), message, time_message))
    sys.stdout.flush()

    # Final call
    if p == length:
        total = int(time.time()-start_time)
        if total > 60:
            time_message = ' ' + str(total//60) + 'min'
        else:
            time_message = ' ' + str(total) + 'sec'
        
        final_offset = TEXTWIDTH-len(time_message)
        final_message = string.ljust(final_message, final_offset)[:final_offset]
        sys.stdout.write('\r{0}{1}'.format(final_message, time_message))
        sys.stdout.flush()
        start_time = None
        stop_time = None
        print("")
        
if __name__ == "__main__":
#    fully_sample_sky(region = "allsky")
    #gauss_sample_sky(region = "allsky", useprior = "ThetaRHT")
    #gauss_sample_region(local = False)
    #fully_sample_sky(region = "allsky", useprior = "RHTPrior", velrangestring = "-4_3", gausssmooth_prior = False)
    #fully_sample_sky(region = "allsky", useprior = "RHTPrior", velrangestring = "-4_3", gausssmooth_prior = True)
    #fully_sample_sky(region = "allsky", limitregion = True, useprior = "RHTPrior", velrangestring = "-4_3", gausssmooth_prior = False)
    #fully_sample_sky(region = "allsky", limitregion = True, adaptivep0 = True, useprior = "RHTPrior", velrangestring = "-4_3", gausssmooth_prior = True)
    #fully_sample_planck_sky(region = "allsky", limitregion = False)
    
    fully_sample_planck_sky(region = "allsky", adaptivep0 = True, limitregion = True, local = False, verbose = False)
    """
    allskypmb = hp.fitsfunc.read_map("/disks/jansky/a/users/goldston/susan/Wide_maps/pMB_DR2_SC_241_353GHz_take2.fits")
    allskypsimb = hp.fitsfunc.read_map("/disks/jansky/a/users/goldston/susan/Wide_maps/psiMB_DR2_SC_241_353GHz_take2.fits")
    
    allskypmb_nest = hp.pixelfunc.reorder(allskypmb, r2n=True)
    allskypsimb_nest = hp.pixelfunc.reorder(allskypsimb, r2n=True)
    
    hpnums = [3785126, 1966514]
    for hpnum in hpnums:
        print("psi is ", allskypsimb_nest[hpnum])
        print("p is ", allskypmb_nest[hpnum])
    """
    
    
    