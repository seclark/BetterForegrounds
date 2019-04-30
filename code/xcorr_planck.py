from __future__ import division
import numpy as np
import healpy as hp
import h5py
import pymaster as nmt
import glob
import os

import xcorr_namaster as xm

# load Planck data
I353, Q353, U353 = xm.get_planck_data(nu=353, local=False, QU=False, IQU=True)
I857, Q857, U857 = xm.get_planck_data(nu=857, local=False, QU=False, IQU=True, vers="R3.01")

#xm.xcorr_TEB(I353, Q353, U353, I857, Q857, U857, apod_mask=mask_apod, bins=bins, nside=nside, 
#          savedata=True, EBpure=True, dataname=["planck353", "217"], savestr="flatpriorplanck", verbose=1, data_root="../data/", **dict_kwargs)
