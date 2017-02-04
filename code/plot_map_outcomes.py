from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


# delta function test data
root = "/Volumes/DataDavy/Foregrounds/BayesianMaps/"
pdeltafunc_fn = "pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"  
psideltafunc_fn = "psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"

# legit process data
pMLmap_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
psiMLmap_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"

def get_nonzero_data(hp_data_fn, mask=None):

    data = hp.fitsfunc.read_map(hp_data_fn)

    if mask is not None:
        data = data[mask]
    else:
        data[np.where(data == 0)] = None
        data = data[~np.isnan(data)]
    
    return data
    
def plot_psi_p_hists(*plotdata, **kwargs):
    
    fig = plt.figure(facecolor="white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax = [ax1, ax2]
    
    for i, data in enumerate(plotdata):
        ax[i].hist(data[0], **kwargs)
        ax[i].hist(data[1], **kwargs)
        
p1 = get_nonzero_data(root + pdeltafunc_fn)
psi1 = get_nonzero_data(root + psideltafunc_fn)

p2 = get_nonzero_data(pMLmap_fn)
psi2 = get_nonzero_data(psiMLmap_fn)

histkwargs = {'bins': 100, 'histtype': 'step'}
plot_psi_p_hists([p1, psi1], [p2, psi2] **histkwargs)
