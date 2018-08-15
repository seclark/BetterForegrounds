import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
import os

def make_mask(nside, GALFA_cut=False, b_cut=False, save_mask=False):
    """
    make a mask.
    GALFA_cut : conservative cut on GALFA DEC range: 1-37
    b_cut     : cut on |b|
    """
    
    out_fn = "../data/masks/mask_absbcut_{}_GALFAcut_{}.fits".format(b_cut, GALFA_cut)
    if os.path.isfile(out_fn):
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
    Cl_A_B = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Bfield)) 
    Cl_A_A = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Afield)) 
    Cl_B_B = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_Bfield, EB_Bfield)) 
    
    if savedata:
        data_root = "../data/"
        Aname = dataname[0]
        Bname = dataname[1]
        out_fn = data_root + "Cl_{}_{}_EBpure_{}_{}{}.h5".format(Aname, Bname, EBpure, nside, savestr)
        
        with h5py.File(outfn, 'w') as f:
            dset = f.create_dataset(name='Cl_A_B', data=Cl_A_B)
            dset = f.create_dataset(name='Cl_A_A', data=Cl_A_A)
            dset = f.create_dataset(name='Cl_B_B', data=Cl_B_B)
            dset.attrs['nside'] = nside
            dset.attrs['EBpure'] = EBpure
            dset.attrs['bins'] = bins
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
    mask_b30 = make_mask(nside, GALFA_cut=True, b_cut=30, save_mask=True)
    mask_b30_apod = apodize_mask(mask_b30, apod_arcmin=60, apod_type='C2')

    # pure
    EB_353_pure = nmt.NmtField(mask_b30_apod, [Q353, U353], purify_e = True, purify_b = True)
    EB_217_pure = nmt.NmtField(mask_b30_apod, [Q217, U217], purify_e = True, purify_b = True)

    # define workspaces
    #w_nonpure = nmt.NmtWorkspace()
    w_pure  = nmt.NmtWorkspace()
    
    # define bins
    bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)

    w_pure.compute_coupling_matrix(EB_353_pure, EB_217_pure, bins)

    # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    Cl_353_217_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_353_pure, EB_217_pure)) 
    Cl_EE_353_217 = Cl_353_217_pure[0]
    Cl_EB_353_217 = Cl_353_217_pure[1]
    Cl_BB_353_217 = Cl_353_217_pure[3]
    
    np.save("../data/Cl_353_217_pure_test.npy", Cl_353_217_pure)
    
    # Same thing for autocorrelations
    w_pure.compute_coupling_matrix(EB_353_pure, EB_353_pure, bins)
    
    Cl_353_353_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_353_pure, EB_353_pure)) 
    Cl_EE_353_353 = Cl_353_353_pure[0]
    Cl_EB_353_353 = Cl_353_353_pure[1]
    Cl_BB_353_353 = Cl_353_353_pure[3]
    
    w_pure.compute_coupling_matrix(EB_217_pure, EB_217_pure, bins)
    
    Cl_217_217_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(EB_217_pure, EB_217_pure)) 
    Cl_EE_217_217 = Cl_217_217_pure[0]
    Cl_EB_217_217 = Cl_217_217_pure[1]
    Cl_BB_217_217 = Cl_217_217_pure[3]
    
    np.save("../data/Cl_353_353_pure_test.npy", Cl_353_353_pure)
    np.save("../data/Cl_217_217_pure_test.npy", Cl_217_217_pure)








        