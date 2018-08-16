import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import os
import h5py

def make_mask(nside, GALFA_cut=False, b_cut=False, save_mask=False):
    """
    make a mask.
    GALFA_cut : conservative cut on GALFA DEC range: 1-37
    b_cut     : cut on |b|
    """
    
    out_fn = "../data/masks/mask_absbcut_{}_GALFAcut_{}.fits".format(b_cut, GALFA_cut)
    if os.path.isfile(out_fn):
        print("Loading mask {}".format(out_fn))
        mask = hp.fitsfunc.read_map(out_fn, nest=False)
        return mask
    
    npix = 12*nside**2
    mask = np.zeros(npix, np.int_)
    
    # coordinates of every pixel
    all_l, all_b = hp.pixelfunc.pix2ang(nside, np.arange(npix), lonlat=True)
    all_coords = SkyCoord(frame="galactic", l=all_l*u.deg, b=all_b*u.deg)
    all_coords_icrs = all_coords.icrs
    all_ra  = all_coords_icrs.ra.deg
    all_dec = all_coords_icrs.dec.deg
    
    if GALFA_cut:
        mask[np.where((all_dec < 37.0) & (all_dec > 1.0))] = 1
    
    if b_cut:
        mask[np.where(np.abs(all_b) < b_cut)] = 0
        
    if save_mask:
        hp.fitsfunc.write_map(out_fn, mask, nest=False, fits_IDL=False, coord='G')
        
    return mask
    
def apodize_mask(mask, apod_arcmin=60, apod_type='C2'):
    """
    for pure-B formalism, must be differentiable at boundary : use 'C1' or 'C2' schemes
    """
    
    apod_deg = apod_arcmin/60.
    mask_apod = nmt.mask_apodization(mask, apod_deg, apotype=apod_type)

    return mask_apod
    
def get_planck_data(nu=353, local=False, QU=True, IQU=False):
    """
    currently loads R3 data
    nu  : frequency in GHz
    IQU : just load IQU instead of all data
    """
    if local:
        planck_root = "/Users/susanclark/Dropbox/Planck/"
    else:
        planck_root = "/data/seclark/Planck/"
    if nu == 353:
        nustr = "-psb"
    else:
        nustr = ""
    planck_fn = planck_root + "HFI_SkyMap_{}{}_2048_R3.00_full.fits".format(nu, nustr)
    
    if QU:
        read_fields = (1,2)
    elif IQU:
        read_fields = (0,1,2)
    else:
        read_fields = (0,1,2,3,4,5,6,7,8,9)
        
    out_data = hp.fitsfunc.read_map(planck_fn, field=read_fields)
    
    if IQU:
        I_data = out_data[0]
        Q_data = out_data[1]
        U_data = out_data[2]
    
        return I_data, Q_data, U_data
        
    if QU:
        Q_data = out_data[0]
        U_data = out_data[1]  
        
        return Q_data, U_data  
    
    else:
        print("Need to implement all-fields read")
        
def make_bins(nside=2048, binwidth=20, ellmax=1001):
    bins = nmt.NmtBin(nside, nlb=binwidth, lmax=int(ellmax))
    ell_binned = bins.get_effective_ells()
    nbins = len(ell_binned)
    
    return bins, ell_binned
    
def xcorr_E_B(Q_Afield, U_Afield, Q_Bfield, U_Bfield, apod_mask=None, bins=None, nside=2048, savedata=True, EBpure=True, dataname=["A", "B"], savestr="", **kwargs):
    """
    Cross- and autocorrelations between two fields.
    
    savedata : make hdf5 file containing Cl outputs and useful parameters.
    """
    
    if EBpure:
        purify_e = True
        purify_b = True
        
    EB_Afield = nmt.NmtField(apod_mask, [Q_Afield, U_Afield], purify_e=purify_e, purify_b=purify_b)
    EB_Bfield = nmt.NmtField(apod_mask, [Q_Bfield, U_Bfield], purify_e=purify_e, purify_b=purify_b)

    # define workspace
    w = nmt.NmtWorkspace()
    
    if bins == None:
        bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)
    else:
        ell_binned = bins.get_effective_ells()
    
    # Mode coupling matrix depends only on masks, not actual fields, so don't need to do this again for autocorr
    w.compute_coupling_matrix(EB_Afield, EB_Bfield, bins)

    # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    Cl_A_B = w.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Bfield)) 
    Cl_A_A = w.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Afield)) 
    Cl_B_B = w.decouple_cell(nmt.compute_coupled_cell(EB_Bfield, EB_Bfield)) 
    
    if savedata:
        data_root = "../data/"
        Aname = dataname[0]
        Bname = dataname[1]
        out_fn = data_root + "Cl_{}_{}_EBpure_{}_{}{}.h5".format(Aname, Bname, EBpure, nside, savestr)
        print("Saving data to {}".format(out_fn))
        
        with h5py.File(out_fn, 'w') as f:
            dset = f.create_dataset(name='Cl_A_B', data=Cl_A_B)
            dset1= f.create_dataset(name='Cl_A_A', data=Cl_A_A)
            dset2= f.create_dataset(name='Cl_B_B', data=Cl_B_B)
            dset.attrs['nside'] = nside
            dset.attrs['EBpure'] = EBpure
            dset.attrs['ell_binned'] = ell_binned
            
            # add arbitrary kwargs as attributes
            for key in kwargs.keys():
                dset.attrs[key] = kwargs[key]
    
    else:
        return Cl_A_B, Cl_A_A, Cl_B_B
    
if __name__ == "__main__":
    
    Q353, U353 = get_planck_data(nu=353, local=False, QU=True, IQU=False)
    Q217, U217 = get_planck_data(nu=217, local=False, QU=True, IQU=False)

    nside = 2048
    
    # mask parameters
    b_cut = 30
    GALFA_cut = False
    apod_arcmin = 60
    apod_type = 'C2'
    mask_b30 = make_mask(nside, GALFA_cut=GALFA_cut, b_cut=b_cut, save_mask=True)
    mask_b30_apod = apodize_mask(mask_b30, apod_arcmin=apod_arcmin, apod_type=apod_type)
    
    # define bins
    bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)
    
    # pass extra kwargs to be saved with data as hdf5 attributes
    dict_kwargs = {"mask_bcut": b_cut, "mask_GALFA_cut": GALFA_cut, 'mask_apod_arcmin': apod_arcmin, 'mask_apod_type': apod_type}
    
    # define a filename string based on these keys
    outstr = ""
    for _key in dict_kwargs.keys():
        outstr += "{}_{}_".format(_key, dict_kwargs[_key])    
    outstr += "test"
    
    xcorr_E_B(Q353, U353, Q217, U217, apod_mask=mask_b30_apod, bins=bins, nside=2048, 
              savedata=True, EBpure=True, dataname=["353", "217"], savestr=outstr, **dict_kwargs)

    




        