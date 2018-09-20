from __future__ import division
import numpy as np
import healpy as hp
import h5py
import pymaster as nmt
import glob
import os

import xcorr_namaster as xm

def get_map(testname):
    bayesroot = "/data/seclark/BetterForegrounds/data/BayesianMaps/"
    mapfn=bayesroot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_planckpatch_{}.fits".format(testname)
    mapI, mapQ, mapU = hp.fitsfunc.read_map(mapfn, field=(0,1,2))
    
    return mapI, mapQ, mapU
    
def get_planck_flatprior_map(testname="newplanck"):
    bayesroot = "/data/seclark/BetterForegrounds/data/BayesianMaps/"
    mapfn=bayesroot+"HFI_SkyMap_353_2048_R2.02_full_pMB_psiMB_{}.fits".format(testname)
    mapI, mapQ, mapU = hp.fitsfunc.read_map(mapfn, field=(0,1,2))
    
    return mapI, mapQ, mapU

if __name__ == "__main__":

    nside=2048
    GALFA_cut = True
    b_cut = 30
    mask = xm.make_mask(nside, GALFA_cut=GALFA_cut, b_cut=b_cut, save_mask=False)
    apod_arcmin = 60
    apod_type = 'C2'
    mask_apod = xm.apodize_mask(mask, apod_arcmin=apod_arcmin, apod_type=apod_type)
    print("mask apodized. Mask shape {}".format(mask_apod.shape))

    # define bins
    bins, ell_binned = xm.make_bins(nside=nside, binwidth=20, ellmax=3001)

    # pass extra kwargs to be saved with data as hdf5 attributes
    dict_kwargs = {'GALFA_cut': GALFA_cut, 'b_cut': b_cut, 'apod': apod_arcmin, 'type': apod_type}

    #mapI, mapQ, mapU = get_map("weight0")
    mapI, mapQ, mapU = get_planck_flatprior_map("newplanck")
    I217, Q217, U217 = xm.get_planck_data(nu=217, local=False, QU=False, IQU=True)
    xm.xcorr_TEB(mapI, mapQ, mapU, I217, Q217, U217, apod_mask=mask_apod, bins=bins, nside=nside, 
              savedata=True, EBpure=True, dataname=["planck353", "217"], savestr="flatpriorplanck", verbose=1, data_root="../data/", **dict_kwargs)






