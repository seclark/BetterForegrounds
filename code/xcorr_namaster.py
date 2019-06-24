from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.io import fits
import os
import h5py

def make_mask(nside, GALFA_cut=False, b_cut=False, save_mask=False, nonabsb_cut=False, nonabsb_cut_gt=False):
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
    
    if nonabsb_cut and not GALFA_cut:
        print("non abs b cut but no GALFA cut")
        print("Setting latitudes below {} to 0".format(nonabsb_cut))
        mask = np.ones(npix, np.int_)
        mask[np.where(all_b < nonabsb_cut)] = 0

    if nonabsb_cut_gt and not GALFA_cut:
        print("non abs b cut but no GALFA cut")
        print("Setting latitudes greater than {} to 0".format(nonabsb_cut_gt))
        mask = np.ones(npix, np.int_)
        mask[np.where(all_b > nonabsb_cut_gt)] = 0
        
    if b_cut and not GALFA_cut:
        print("b cut but no GALFA cut")
        mask = np.ones(npix, np.int_)
        mask[np.where(np.abs(all_b) < b_cut)] = 0
        
    if save_mask:
        hp.fitsfunc.write_map(out_fn, mask, nest=False, fits_IDL=False, coord='G')
        
    return mask
    
def make_circular_pixel_mask(nside=1024, ipix=4000000, raddeg=2):
    
    mask = np.zeros(12*nside**2)
    mask[hp.query_disc(nside, hp.pix2vec(nside, ipix), np.radians(raddeg), nest=False)] = 1
    
    return mask

def load_Planck_mask(skycoverage=70, nside=2048, local=False):
    """
    Load one of the Planck-provided masks
    """
    
    if local:
        maskroot = "/Users/susanclark/Dropbox/Planck/"
    else:
        maskroot = "/data/seclark/BetterForegrounds/data/masks/"
    maskhdu=fits.open(maskroot+"HFI_Mask_GalPlane-apo0_2048_R2.00.fits")
    maskstr = "GAL{0:0=3d}".format(skycoverage)
    masknest = maskhdu[1].data[maskstr]
    maskring = hp.pixelfunc.reorder(masknest, n2r=True)
    
    if nside != 2048:
        # going straight from 2048 to 64 breaks things, so:
        if nside <= 64:
            maskring_lr = hp.ud_grade(maskring, 512)
            maskring = hp.ud_grade(maskring_lr, nside)
            print("successfully downgraded to nside {}".format(nside))
        else:
            maskring = hp.ud_grade(maskring, nside)
            print("successfully downgraded to nside {}".format(nside))
    
    return maskring
    
def apodize_mask(mask, apod_arcmin=60, apod_type='C2'):
    """
    for pure-B formalism, must be differentiable at boundary : use 'C1' or 'C2' schemes
    """
    
    apod_deg = apod_arcmin/60.
    mask_apod = nmt.mask_apodization(mask, apod_deg, apotype=apod_type)

    return mask_apod
    
def get_planck_data(nu=353, local=False, Ionly=False, QU=False, IQU=False, vers="R3.01"):
    """
    currently loads R3 data in RING ordering.
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
    if vers == "R3.00": 
        planck_fn = planck_root + "HFI_SkyMap_{}{}-field-IQU_2048_R3.00_full.fits".format(nu, nustr)
    else:
        planck_fn = planck_root + "HFI_SkyMap_{}{}_2048_{}_full.fits".format(nu, nustr, vers)
    print("loading {}".format(planck_fn))
    
    if QU:
        read_fields = (1,2)
    elif IQU:
        read_fields = (0,1,2)
    else:
        read_fields = (0,1,2,3,4,5,6,7,8,9)
    if Ionly:
        read_fields = 0
        
    out_data = hp.fitsfunc.read_map(planck_fn, field=read_fields, nest=False)
    
    if IQU:
        I_data = out_data[0]
        Q_data = out_data[1]
        U_data = out_data[2]
    
        return I_data, Q_data, U_data
        
    elif QU:
        Q_data = out_data[0]
        U_data = out_data[1]  
        
        return Q_data, U_data  
        
    elif Ionly:
        return out_data
    
    else:
        print("Need to implement all-fields read")

        
def make_bins(nside=2048, binwidth=20, ellmax=1001):
    bins = nmt.NmtBin(nside, nlb=binwidth, lmax=int(ellmax))
    ell_binned = bins.get_effective_ells()
    nbins = len(ell_binned)
    
    return bins, ell_binned

def xcorr_TEB(I_Afield, Q_Afield, U_Afield, I_Bfield, Q_Bfield, U_Bfield, apod_mask=None, bins=None, nside=2048, savedata=True, EBpure=True, dataname=["A", "B"], savestr="", verbose=0, data_root="../data/", **kwargs):
    print("Starting.")
    
    if EBpure:
        purify_e = True
        purify_b = True
        
    # spin 2 fields
    EB_Afield = nmt.NmtField(apod_mask, [Q_Afield, U_Afield], purify_e=purify_e, purify_b=purify_b)
    EB_Bfield = nmt.NmtField(apod_mask, [Q_Bfield, U_Bfield], purify_e=purify_e, purify_b=purify_b) 
    # spin 0 fields  
    T_Afield =  nmt.NmtField(apod_mask, [I_Afield])
    T_Bfield =  nmt.NmtField(apod_mask, [I_Bfield])
    
    if verbose:
        print("Computed TEB for both fields")
    
    # define workspace
    w = nmt.NmtWorkspace()
    
    if verbose:
        print("Workspace ready")
    
    if bins == None:
        bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)
    else:
        ell_binned = bins.get_effective_ells()
        
    #Compute MASTER estimator
    #spin-0 x spin-0
    ClAA_00=nmt.compute_full_master(T_Afield, T_Afield, bins)
    ClAB_00=nmt.compute_full_master(T_Afield, T_Bfield, bins)
    ClBB_00=nmt.compute_full_master(T_Bfield, T_Bfield, bins)
    #spin-0 x spin-2
    ClAA_02=nmt.compute_full_master(T_Afield, EB_Afield, bins)
    ClAB_02=nmt.compute_full_master(T_Afield, EB_Bfield, bins)
    ClBB_02=nmt.compute_full_master(T_Bfield, EB_Bfield, bins)
    #spin-2 x spin-2
    ClAA_22=nmt.compute_full_master(EB_Afield, EB_Afield, bins)
    ClAB_22=nmt.compute_full_master(EB_Afield, EB_Bfield, bins)
    ClBB_22=nmt.compute_full_master(EB_Bfield, EB_Bfield, bins)
        
    if verbose:
        print("Data ready to be saved")
    
    if savedata:
        Aname = dataname[0]
        Bname = dataname[1]
        out_fn = data_root + "Cl_{}_{}_TEB_EBpure_{}_{}_{}.h5".format(Aname, Bname, EBpure, nside, savestr)
        print("Saving data to {}".format(out_fn))
        
        with h5py.File(out_fn, 'w') as f:
            dset = f.create_dataset(name='ClAB_00', data=ClAB_00)
            dset1= f.create_dataset(name='ClAA_00', data=ClAA_00)
            dset2= f.create_dataset(name='ClBB_00', data=ClBB_00)
            
            dset3= f.create_dataset(name='ClAB_02', data=ClAB_02)
            dset4= f.create_dataset(name='ClAA_02', data=ClAA_02)
            dset5= f.create_dataset(name='ClBB_02', data=ClBB_02)
        
            dset6= f.create_dataset(name='ClAB_22', data=ClAB_22)
            dset7= f.create_dataset(name='ClAA_22', data=ClAA_22)
            dset8= f.create_dataset(name='ClBB_22', data=ClBB_22)
            dset.attrs['nside'] = nside
            dset.attrs['EBpure'] = EBpure
            dset.attrs['ell_binned'] = ell_binned
            
            # add arbitrary kwargs as attributes
            for key in kwargs.keys():
                dset.attrs[key] = kwargs[key]
                
                
def xcorr_T_EB(I_Afield, Q_Bfield, U_Bfield, apod_mask=None, bins=None, nside=2048, savedata=True, EBpure=True, CFM=False, Cerrors=False, dataname=["A", "B"], savestr="", verbose=0, data_root="../data/", **kwargs):
    print("Starting.")
    
    if EBpure:
        purify_e = True
        purify_b = True
        if verbose:
            print("Purifying E and B")
    
    print("fsky = {}".format(np.sum(mask_apod)/len(mask_apod)))
    print("Q_Bfield: ", Q_Bfield.shape, Q_Bfield.dtype)
    
    # spin 2 fields
    EB_Bfield = nmt.NmtField(apod_mask, [Q_Bfield, U_Bfield], purify_e=purify_e, purify_b=purify_b) 
    if verbose:
        print("spin 2 done, now to spin 0")
    # spin 0 fields  
    T_Afield =  nmt.NmtField(apod_mask, [I_Afield])
    
    if verbose:
        print("Computed TEB for both fields")
    
    # define workspace
    w = nmt.NmtWorkspace()
    
    if verbose:
        print("Workspace ready")
    
    if bins == None:
        bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)
    else:
        ell_binned = bins.get_effective_ells()
        
    if CFM:
        if verbose:
            print("Compute full master")
            
        #Compute MASTER estimator
        #spin-0 x spin-2
        ClAB_02=nmt.compute_full_master(T_Afield, EB_Bfield, bins)
        
        if Cerrors:
            #spin-0 x spin-0
            ClAA_00=nmt.compute_full_master(T_Afield, T_Afield, bins)
            #spin-2 x spin-2
            ClBB_22=nmt.compute_full_master(EB_Bfield, EB_Bfield, bins)
    else:
        w.compute_coupling_matrix(T_Afield, EB_Bfield, bins)
        
        if verbose:
            print("Mode coupling matrix computed")

        # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
        ClAB_02 = w.decouple_cell(nmt.compute_coupled_cell(T_Afield, EB_Bfield)) 
        
        if Cerrors:
            print("errors not implementsed for non CFM")
            
    if Cerrors:
        #error = sqrt( TT*BB * (1/fsky) * (1/(2*ell + 1)) * (1/ellbinwidth) )
        TT = ClAA_00[0]
        EE = ClBB_22[0]
        BB = ClBB_22[3]
        fsky = np.sum(apod_mask/len(apod_mask))
        ell_binwidth = ell_binned[1] - ell_binned[0] # assumes constant
        err_TE = (TT*EE) / (fsky*(2*ell_binned + 1)*ell_binwidth) 
        err_TB = (TT*BB) / (fsky*(2*ell_binned + 1)*ell_binwidth) 
        
    if verbose:
        print("Data ready to be saved")
    
    if savedata:
        Aname = dataname[0]
        Bname = dataname[1]
        out_fn = data_root + "Cl_{}_{}_TEB_EBpure_{}_{}_{}.h5".format(Aname, Bname, EBpure, nside, savestr)
        print("Saving data to {}".format(out_fn))
        
        with h5py.File(out_fn, 'w') as f:
            #dset1= f.create_dataset(name='ClAA_00', data=ClAA_00)
            
            dset = f.create_dataset(name='ClAB_02', data=ClAB_02)
            if Cerrors:
                TT = f.create_dataset(name='TT', data=TT)
                EE = f.create_dataset(name='EE', data=EE)
                BB = f.create_dataset(name='BB', data=BB)
                
                errTE = f.create_dataset(name='err_TE', data=err_TE)
                errTB = f.create_dataset(name='err_TB', data=err_TB)
            
            #dset8= f.create_dataset(name='ClBB_22', data=ClBB_22)
            dset.attrs['nside'] = nside
            dset.attrs['EBpure'] = EBpure
            dset.attrs['ell_binned'] = ell_binned
            
            # add arbitrary kwargs as attributes
            for key in kwargs.keys():
                dset.attrs[key] = kwargs[key]
    

    
def xcorr_E_B(Q_Afield, U_Afield, Q_Bfield, U_Bfield, apod_mask=None, bins=None, nside=2048, savedata=True, EBpure=True, dataname=["A", "B"], savestr="", verbose=0, data_root="../data/", **kwargs):
    """
    Cross- and autocorrelations between two fields.
    
    savedata : make hdf5 file containing Cl outputs and useful parameters.
    """
    print("Starting.")
    
    if EBpure:
        purify_e = True
        purify_b = True
        
    EB_Afield = nmt.NmtField(apod_mask, [Q_Afield, U_Afield], purify_e=purify_e, purify_b=purify_b)
    EB_Bfield = nmt.NmtField(apod_mask, [Q_Bfield, U_Bfield], purify_e=purify_e, purify_b=purify_b)
    
    if verbose:
        print("Computed EB for both fields")

    # define workspace
    w = nmt.NmtWorkspace()
    
    if verbose:
        print("Workspace ready")
    
    if bins == None:
        bins, ell_binned = make_bins(nside=nside, binwidth=20, ellmax=1001)
    else:
        ell_binned = bins.get_effective_ells()
    
    # Mode coupling matrix depends only on masks, not actual fields, so don't need to do this again for autocorr
    w.compute_coupling_matrix(EB_Afield, EB_Bfield, bins)
    
    if verbose:
        print("Mode coupling matrix computed")

    # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    Cl_A_B = w.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Bfield)) 
    Cl_A_A = w.decouple_cell(nmt.compute_coupled_cell(EB_Afield, EB_Afield)) 
    Cl_B_B = w.decouple_cell(nmt.compute_coupled_cell(EB_Bfield, EB_Bfield)) 
    
    if verbose:
        print("Data ready to be saved")
    
    if savedata:
        Aname = dataname[0]
        Bname = dataname[1]
        out_fn = data_root + "Cl_{}_{}_EBpure_{}_{}_{}.h5".format(Aname, Bname, EBpure, nside, savestr)
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
        
def example_E_B_Planck():
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
    




    




        