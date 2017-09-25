from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import copy

import sys 
sys.path.insert(0, '../../ACTPol/code')
import foreground_tools


def nonzero_data(data, mask=None, samesize=False):

    if samesize:
        datacopy = copy.copy(data)

    if mask is not None:
        data = data[np.where(mask == 1)]
        if samesize:
            datacopy[np.where(mask != 1)] = None
    else:
        data[np.where(data == 0)] = None
        if samesize:
            datacopy[np.where(data == 0)] = None
    
    data = data[~np.isnan(data)]
    
    if samesize:
        return data, datacopy
    else:    
        return data

def get_nonzero_data(hp_data_fn, mask=None, samesize=False):

    data = hp.fitsfunc.read_map(hp_data_fn)

    if samesize:
        data, datacopy = nonzero_data(data, mask=mask, samesize=samesize)
        return data, datacopy
        
    else:
        data = nonzero_data(data, mask=mask, samesize=samesize)
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
        
if __name__ == "__main__":

    # delta function test data
    root = "/Volumes/DataDavy/Foregrounds/BayesianMaps/"
    pdeltafunc_fn = "pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"  
    psideltafunc_fn = "psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True.fits"

    # delta func #2 where I didn't convert to pixels
    #pdeltafunc_fn2 = "pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True_2.fits"  
    #psideltafunc_fn2 = "psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_False_deltafuncprior_True_2.fits"

    # legit process data
    pMLmap_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"
    psiMLmap_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_-4_3_smoothprior_True_adaptivep0_True_new.fits"

    # planck angles from flat-prior
    p_planck_flatprior_fn = "/Users/susanclark/BetterForegrounds/data/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
    psi_planck_flatprior_fn = "/Users/susanclark/BetterForegrounds/data/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"

    # hp projected raw angles
    #QRHTmap = "Q_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected.fits"

    #QRHT, URHT, PRHT, theta_rht, int_rhtunsmoothed, QRHTsq, URHTsq = foreground_tools.get_QU_RHT_corrected(region = "SC_241", wlen = 75, smr = 15, smoothRHT = False, sigma = 0, QUmean = False, bwrm = True, galfapixcorr = True, intRHTcorr = False)
    #Q, U, Pplanck, psi = foreground_tools.get_QU_corrected(region = "SC_241", smoothPlanck = False, sigma = 0, QUmean = False)

    # Planck angles in original form
    planck353_fn = "/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits"
    planckT, planckQ, planckU = hp.fitsfunc.read_map(planck353_fn, field=(0,1,2))
    psi = np.mod(0.5*np.arctan2(planckU, planckQ), np.pi) #Everything in Planck (non-IAU), Galactic, pol angle coords

    # RHT angles in *galactic* coordinates
    RHT_TQU_Gal_fn = "../data/TQU_RHT_SC_241_best_ch16_to_24_w75_s15_t70_bwrm_galfapixcorr_UPSIDEDOWN_hp_projected_Equ_inGal_mask.fits"
    RHT_T_Gal, QRHT, URHT = hp.fitsfunc.read_map(RHT_TQU_Gal_fn, field=(0,1,2))
    intrhtmask = np.zeros(RHT_T_Gal.shape)
    intrhtmask[RHT_T_Gal > 0.5] = 1

    theta_rht = np.mod(0.5*np.arctan2(URHT, -QRHT), np.pi) # from IAU B-field -> nonIAU pol angle

    #intrhtmask = np.zeros(int_rhtunsmoothed.shape)
    #intrhtmask[np.where(int_rhtunsmoothed > 0)] = 1

    p1 = get_nonzero_data(root + pdeltafunc_fn)
    psi1 = get_nonzero_data(root + psideltafunc_fn)

    #p2 = get_nonzero_data(root + pdeltafunc_fn2)
    #psi2 = get_nonzero_data(root + psideltafunc_fn2)

    p3 = get_nonzero_data(pMLmap_fn)
    psi3, psi3plot = get_nonzero_data(psiMLmap_fn, samesize=True)

    pflat = get_nonzero_data(p_planck_flatprior_fn)
    psiflat, psiflatplot = get_nonzero_data(psi_planck_flatprior_fn, samesize=True)

    planckpsi, planckpsiplot = nonzero_data(psi, mask=intrhtmask, samesize=True)

    rhtpsi, rhtpsiplot = nonzero_data(theta_rht, mask=intrhtmask, samesize=True)

    histkwargs = {'bins': 100, 'histtype': 'step'}
    #plot_psi_p_hists([p1, psi1, 'delta func', 'red'], [p2, psi2, 'RHT', 'teal'], [p3, psi3, 'delta 2', 'orange'], **histkwargs)

    #plot_psi_hists([psi1, 'delta func', 'red'], [psi3, 'RHT prior', 'teal'], [planckpsi, 'planck orig', 'orange'], [rhtpsi, 'RHT orig', 'blue'], **histkwargs)
    plot_psi_hists([psiflat, 'planck flat', 'red'], [psi3, 'RHT prior', 'teal'], [planckpsi, 'planck orig', 'orange'], [rhtpsi, 'RHT orig', 'blue'], **histkwargs)


