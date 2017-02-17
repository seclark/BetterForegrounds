from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


import sys 
sys.path.insert(0, '../../ACTPol/code')
import foreground_tools

# delta function test data
root = "/Volumes/DataDavy/Foregrounds/BayesianMaps/"
pdeltafunc_fn = "pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"  
psideltafunc_fn = "psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"

# delta func #2 where I didn't convert to pixels
pdeltafunc_fn2 = "pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True_2.fits"  
psideltafunc_fn2 = "psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True_2.fits"

# legit process data
pMLmap_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
psiMLmap_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"

QRHT, URHT, PRHT, theta_rht, int_rhtunsmoothed, QRHTsq, URHTsq = foreground_tools.get_QU_RHT_corrected(region = "SC_241", wlen = 75, smr = 15, smoothRHT = False, sigma = 0, QUmean = False, bwrm = True, galfapixcorr = True, intRHTcorr = False)
Q, U, Pplanck, psi = foreground_tools.get_QU_corrected(region = "SC_241", smoothPlanck = False, sigma = 0, QUmean = False)

intrhtmask = np.zeros(int_rhtunsmoothed.shape)
intrhtmask[np.where(int_rhtunsmoothed > 0)] = 1

def nonzero_data(data, mask=None):

    if mask is not None:
        data = data[np.where(mask == 1)]
    else:
        data[np.where(data == 0)] = None
    
    data = data[~np.isnan(data)]
    
    return data

def get_nonzero_data(hp_data_fn, mask=None):

    data = hp.fitsfunc.read_map(hp_data_fn)

    data = nonzero_data(data, mask=mask)
    
    return data
    
def plot_psi_p_hists(*plotdata, **kwargs):
    
    fig = plt.figure(facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax = [ax1, ax2]
    
    for i, data in enumerate(plotdata):
        ax[0].hist(data[0], label=data[2], color=data[3], **kwargs)
        ax[1].hist(data[1], label=data[2], color=data[3], **kwargs)
    
    ax1.set_title('p')
    ax2.set_title('psi')    
        
    plt.legend(loc=4)
    
def plot_psi_hists(*plotdata, **kwargs):
    
    fig = plt.figure(facecolor="white")
    ax1 = fig.add_subplot(111)
    
    for i, data in enumerate(plotdata):
        ax1.hist(data[0], label=data[1], color=data[2], **kwargs)
    
    ax1.set_title('psi')  
        
    plt.legend(loc=4)
        
p1 = get_nonzero_data(root + pdeltafunc_fn)
psi1 = get_nonzero_data(root + psideltafunc_fn)

p2 = get_nonzero_data(root + pdeltafunc_fn2)
psi2 = get_nonzero_data(root + psideltafunc_fn2)

p3 = get_nonzero_data(pMLmap_fn)
psi3 = get_nonzero_data(psiMLmap_fn)

planckpsi = nonzero_data(psi, mask=intrhtmask)

rhtpsi = nonzero_data(theta_rht, mask=intrhtmask)

histkwargs = {'bins': 100, 'histtype': 'step'}
#plot_psi_p_hists([p1, psi1, 'delta func', 'red'], [p2, psi2, 'RHT', 'teal'], [p3, psi3, 'delta 2', 'orange'], **histkwargs)

plot_psi_hists([psi1, 'delta func', 'red'], [psi3, 'RHT prior', 'teal'], [planckpsi, 'planck orig', 'orange'], [rhtpsi, 'RHT orig', 'blue'], **histkwargs)
