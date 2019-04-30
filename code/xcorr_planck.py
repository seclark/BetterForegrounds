from __future__ import division
import numpy as np
import healpy as hp
import h5py
import pymaster as nmt
import glob
import os

import xcorr_namaster as xm

if __name__ == "__main__":
    
    # load Planck data
    I353, Q353, U353 = xm.get_planck_data(nu=353, local=False, QU=False, IQU=True)
    I857 = xm.get_planck_data(nu=857, local=False, Ionly=True, QU=False, IQU=False, vers="R3.01")

    # masking
    nside=2048
    mask = xm.make_mask(nside, GALFA_cut=False, b_cut=False, save_mask=False)
    apod_arcmin = 60
    apod_type = 'C2'
    mask_apod = xm.apodize_mask(mask, apod_arcmin=apod_arcmin, apod_type=apod_type)
    print("mask apodized. Mask shape {}".format(mask_apod.shape))
    
    # define bins
    bins, ell_binned = xm.make_bins(nside=nside, binwidth=20, ellmax=1001)

    # pass extra kwargs to be saved with data as hdf5 attributes
    dict_kwargs = {'apod': apod_arcmin, 'type': apod_type}

    xm.xcorr_T_EB(I857, Q353, U353, apod_mask=mask_apod, bins=bins, nside=nside, 
              savedata=True, EBpure=True, dataname=["planck857", "planck353"], savestr="planck857x353", verbose=1, data_root="../data/", **dict_kwargs)
    
    
