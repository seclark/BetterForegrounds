from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import copy
import sys
sys.path.insert(0, '../../PolarizationTools')
import basic_functions as polarization_tools


# load latest attempt
inpath = "/Users/susanclark/BetterForegrounds/data/"
rht_pMLmap=inpath+"pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"
rht_psiMLmap=inpath+"psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"
planck_pMLmap = inpath+"pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
planck_psiMLmap = inpath+"psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"

rht_pML = hp.fitsfunc.read_map(rht_pMLmap)
rht_psiML = hp.fitsfunc.read_map(rht_psiMLmap)

planck_pML = hp.fitsfunc.read_map(planck_pMLmap)
planck_psiML = hp.fitsfunc.read_map(planck_psiMLmap)

fig = plt.figure(facecolor="white")
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

planck_pMLnoz = copy.copy(planck_pML)
planck_pMLnoz[np.where(planck_pML == 0)] = None
planck_psiMLnoz = copy.copy(planck_psiML)
planck_psiMLnoz[np.where(planck_psiML == 0)] = None

rht_pMLnoz = copy.copy(rht_pML)
rht_pMLnoz[np.where(rht_pML == 0)] = None
rht_psiMLnoz = copy.copy(rht_psiML)
rht_psiMLnoz[np.where(rht_psiML == 0)] = None

ax1.hist(rht_pMLnoz[~np.isnan(rht_pMLnoz)], bins=100, histtype = "step", color="teal")
ax2.hist(rht_psiMLnoz[~np.isnan(rht_psiMLnoz)], bins=100, histtype = "step", color="teal")
ax1.hist(planck_pMLnoz[~np.isnan(planck_pMLnoz)], bins=100, histtype = "step", color="orange")
ax2.hist(planck_psiMLnoz[~np.isnan(planck_psiMLnoz)], bins=100, histtype = "step", color="orange")

planck_U = planck_pMLnoz*np.sin(2*planck_psiMLnoz)
planck_Q = planck_pMLnoz*np.cos(2*planck_psiMLnoz)
rht_U = rht_pMLnoz*np.sin(2*rht_psiMLnoz)
rht_Q = rht_pMLnoz*np.cos(2*rht_psiMLnoz)
 
ax3.hist(rht_U[~np.isnan(rht_U)], bins=100, histtype = "step", color="teal")
ax4.hist(rht_Q[~np.isnan(rht_Q)], bins=100, histtype = "step", color="teal")
ax3.hist(planck_U[~np.isnan(planck_U)], bins=100, histtype = "step", color="orange")
ax4.hist(planck_Q[~np.isnan(planck_Q)], bins=100, histtype = "step", color="orange")
ax3.set_title("U")
ax4.set_title("Q")

wherenan = np.zeros(len(planck_psiMLnoz))
wherenan[np.where(np.isnan(planck_pMLnoz) == True)] = None
wherenan[np.where(np.isnan(planck_psiMLnoz) == True)] = None
ang_resid = polarization_tools.angle_residual(planck_psiMLnoz[~np.isnan(wherenan)], rht_psiMLnoz[~np.isnan(wherenan)], degrees=False)

