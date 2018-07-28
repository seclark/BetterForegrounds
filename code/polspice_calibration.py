import numpy as np
import matplotlib
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt
import pyfits
import healpy as hp
import subprocess
import os
from autocorr_parameter_wrapper import *
from mask_apod_automate import *

#####
bash = "bash"
autocorr_script = "/home/jch/Peyton/Dropbox/Thesis_Projects/yNILC/Planck353_angles/BetterForegrounds/code/autocorr_automate.sh"
FITS_end = '.fits'
TXT_end = '.txt'
PDF_end = '.pdf'
DAT_end = '.dat'
#####

#####
# mask parameters
N_side=2048
Npix = 12*N_side**2
Coords = 'G' #coordinate system
mask_dir = '/data/jch/RHT_QU/'
mask_name = 'allsky_GALFA_mask_nonzero_nchannels_edge200_hp_proj_plusbgt70_maskGal'
# apodization parameters (for the tapering function)
FWHM_apod_Arcmin = 15. #FWHM of the gaussian apodization function
# mask file names: boolean and apodized (see mask_apod_automate.py)
mask_file = mask_dir+mask_name+FITS_end
mask_apod_name = mask_name+'_taperFWHM'+str(int(FWHM_apod_Arcmin))+'arcmin'
mask_apod_file = mask_dir+mask_apod_name+FITS_end
fsky_name = mask_name+'_fsky'
fsky_file = fsky_name+TXT_end
#####

#####
# simulation directory, maps, and theory
sim_dir = '/data/jch/Planckdata/DustSims/'
sim_name = 'DustSim_BICEPamp_alpha-2.42_TQU_' #'+str(i)+'.fits'
Nsim = 10 #actually have 100
sim_theory = 'DustEEBB_BICEPamp_alpha-2.42'+TXT_end
EEDust = (np.loadtxt(sim_dir+sim_theory))[:,0] #ClEE -- extends to ell=4000
BBDust = (np.loadtxt(sim_dir+sim_theory))[:,1] #ClBB -- extends to ell=4000
#####

#####
# initialize structure holding all relevant data/info
# this class is defined in autocorr_parameter_wrapper.py
# N.B. no beam in the simulated maps
params = PolParams(apodsigma=7.65, apodtype=0, thetamax=14.0, nlmax=1000, ellmin=40.0, ellmax=1000.0, nbins=6, mapname="", CLAA="", AAOUT="", beamA=0., maskdir=mask_dir, maskname=mask_name, fskyname=fsky_name, Nside=N_side, coords=Coords, FWHM_apod_arcmin=FWHM_apod_Arcmin, CLAAbinned="", kernelfile="", kernelbool=1)
ell = np.arange(int(params.ellmax)+1)
EEDust = EEDust[0:2*int(params.ellmax)+1] #cut to range needed in polspice kernel calculation below
BBDust = BBDust[0:2*int(params.ellmax)+1] #cut to range needed in polspice kernel calculation below
#####

#####
# define the bins (match autocorr_bin.py)
binbounds = np.logspace(np.log10(float(params.ellmin)), np.log10(float(params.ellmax)), num=params.nbins+1, endpoint=True)
binbounds=[np.int(b) for b in binbounds]
Deltaell = np.zeros(params.nbins)
for i in xrange(1,params.nbins+1):
    Deltaell[i-1] = binbounds[i]-binbounds[i-1]
# binning function
def bin_cl(cl,bb):
    ell = np.arange(len(cl))
    # multiply by ell*(ell+1)/2pi before binning
    cltemp = cl*(ell*(ell+1.0))/(2.0*np.pi)
    bb=[np.int(b) for b in bb]
    ellsubarrs = np.split(ell, bb)
    clsubarrs = np.split(cltemp, bb)
    ellbinned = np.zeros(len(bb)+1)
    clbinned = np.zeros(len(bb)+1)
    for i in xrange(len(bb)+1):
        ellbinned[i] = np.mean(ellsubarrs[i])
        clbinned[i] = np.mean(clsubarrs[i])
    # cut out the unwanted elements
    return [ellbinned[1:len(ellbinned)-1], clbinned[1:len(ellbinned)-1]]
#####

#####
# construct the apodized mask
apodize_mask(params.maskdir, params.maskname, params.Nside, params.coords, params.FWHM_apod_arcmin, False)
# update mask name to be the apodized mask
params.maskname = mask_apod_name
#####

#####
# define range of polspice parameters to vary
# specify defaults -- theta_max should be similar to the minimum one-dimensional extent of the mask
#theta_max_default = 14.0
#apod_sigma_default = 7.65
theta_max_MIN = 7.0
theta_max_MAX = 21.0
num_theta_max = 15
theta_max_arr = np.linspace(theta_max_MIN, theta_max_MAX, num=num_theta_max, endpoint=True)
#apod_sigma_MIN = defined adaptively in the loop below
#apod_sigma_MAX = defined adaptively in the loop below
num_apod_sigma = 11 #for each theta_max value
apod_sigma_MIN = theta_max_MIN - 5.0
if (apod_sigma_MIN <= 0.):
    "error: apod_sigma_MIN <= 0."
    quit()
#####

#####
# array to hold chi^2 or other metric
chi2_arr = np.zeros((num_theta_max,num_apod_sigma))
meansqdist_arr = np.zeros((num_theta_max,num_apod_sigma))
# array to hold the apod_sigma values (which are different for each theta_max value)
apod_sigma_arr = np.zeros((num_theta_max,num_apod_sigma))
# loop over polspice parameter variations
for i in xrange(num_theta_max):
    # construct local array of apod_sigma values -- note that the spacing of these  is also different at each theta_max value
    apod_sigma_MAX = theta_max_arr[i]-0.5 #apod_sigma should never be larger than theta_max
    apod_sigma_arr_loc = np.linspace(apod_sigma_MIN, apod_sigma_MAX, num=num_apod_sigma, endpoint=True)
    for j in xrange(num_apod_sigma):
        apod_sigma_arr[i][j] = apod_sigma_arr_loc[j] #save apod_sigma values for later lookup
        # fill out the PolParams structure
        params.apodsigma = apod_sigma_arr_loc[j]
        params.thetamax = theta_max_arr[i]
        # loop over simulations
        # arrays to hold output measured/binned power spectrum of each sim (EE and BB) -- the output file contains columns: ell , EE , BB
        # note that the binning code returns Dl = Cl * ell*(ell+1)/2pi
        DlEE_arr = np.zeros((Nsim,params.nbins))
        DlBB_arr = np.zeros((Nsim,params.nbins))
        for k in xrange(Nsim):
            params.mapname = sim_dir+sim_name+str(k)
            params.CLAA = sim_dir+sim_name+str(k)+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_Cl"+TXT_end
            params.AAOUT = sim_dir+sim_name+str(k)+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_output"+TXT_end
            params.CLAAbinned = sim_dir+sim_name+str(k)+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_DlEEBBbinned"+TXT_end #note name (Dl = ell*(ell+1)/2pi * Cl)
            params.kernelfile = mask_dir+mask_apod_name+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_kernels"+FITS_end
            if (k==0): #only save the kernels when running on the first sim
                params.kernelbool = 1
            else:
                params.kernelbool = 0
            # run polspice and binning code to get ClEE and ClBB for this sim
            subprocess.call([bash, autocorr_script, params.mapname+FITS_end, str(params.apodsigma), str(params.apodtype), str(params.thetamax), str(params.nlmax), params.maskdir+params.maskname+FITS_end, params.CLAA, params.AAOUT, str(params.beamA), params.maskdir+params.fskyname+TXT_end, params.CLAAbinned, params.kernelfile, str(params.ellmin), str(params.ellmax), str(params.nbins), str(params.kernelbool)])
            DlEE_arr[k] = (np.loadtxt(params.CLAAbinned))[:,1]
            DlBB_arr[k] = (np.loadtxt(params.CLAAbinned))[:,2]
        # get the mean and std dev of DlEE and DlBB for each ell bin
        DlEE_mean = np.zeros(params.nbins)
        DlBB_mean = np.zeros(params.nbins)
        DlEE_std  = np.zeros(params.nbins)
        DlBB_std  = np.zeros(params.nbins)
        for b in xrange(params.nbins):
            DlEE_mean[b] = np.mean(DlEE_arr[:,b])
            DlBB_mean[b] = np.mean(DlBB_arr[:,b])
            DlEE_std[b]  = np.std(DlEE_arr[:,b])
            DlBB_std[b]  = np.std(DlBB_arr[:,b])
        # compute theory ClEE and ClBB with polspice kernel and binning
        # polspice kernel file: details given in fits file header
        # COMMENT  E-B Decoupled Estimators                                               
        #COMMENT  <C_TT(l1)>           = Sum_l2 Kern(l1, l2, 1) C_TT(l2)_true            
        #COMMENT  <C_EE(l1)>           = Sum_l2 Kern(l1, l2, 3) C_EE(l2)_true            
        #COMMENT  <C_BB(l1)>           = Sum_l2 Kern(l1, l2, 3) C_BB(l2)_true            
        #COMMENT  <C_TE(l1)>           = Sum_l2 Kern(l1, l2, 4) C_TE(l2)_true            
        #COMMENT  <C_TB(l1)>           = Sum_l2 Kern(l1, l2, 4) C_TB(l2)_true            
        #COMMENT  <C_EB(l1)>           = Sum_l2 Kern(l1, l2, 3) C_EB(l2)_true            
        #COMMENT  with l1 in {0,lmax}, l2 in {0,2*lmax}                                  
        #COMMENT  NB: In the decoupled case, Kern(*,*,2) is *NOT* used                   
        #COMMENT  (see Eq. (91) of Chon et al. 2004)
        hdulist = pyfits.open(params.kernelfile)
        EEkernel = (hdulist[0].data)[2] #size = (2*ellmax+1, ellmax+1)
        BBkernel = EEkernel
        hdulist.close()
        ClEE_theory = np.zeros(int(params.ellmax)+1)
        ClBB_theory = np.zeros(int(params.ellmax)+1)
        for l in xrange(int(params.ellmax)+1):
            ClEE_theory[l] = np.sum( EEkernel[:,l] * EEDust )
            ClBB_theory[l] = np.sum( BBkernel[:,l] * BBDust )
        # save in case useful later
        np.savetxt(sim_dir+sim_theory+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_kernelweighted_Cl"+TXT_end, np.transpose(np.array([ell, ClEE_theory, ClBB_theory])))
        # bin identically to simulation Dl
        [ell_binned, DlEE_theory_binned] = bin_cl(ClEE_theory, binbounds)
        [ell_binned, DlBB_theory_binned] = bin_cl(ClBB_theory, binbounds)
        # save theory
        np.savetxt(sim_dir+sim_theory+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_kernelweighted_binned_Dl"+TXT_end, np.transpose(np.array([ell_binned, DlEE_theory_binned, DlBB_theory_binned])))
        # save sim mean and std dev as well
        np.savetxt(sim_dir+sim_theory+"_thetamax"+str(i)+"_apodsigma"+str(j)+"_sim_mean_stddev_binned_Dl"+TXT_end, np.transpose(np.array([ell_binned, DlEE_mean, DlEE_std, DlBB_mean, DlBB_std])))        
        # compute chi^2 and mean-sq-dist (i.e., no weighting by std dev, in case the chi2 metric favors results with large variances)
        chi2_arr[i][j] = np.sum( (DlEE_mean-DlEE_theory_binned)**2.0 / DlEE_std**2.0 + (DlBB_mean-DlBB_theory_binned)**2.0 / DlBB_std**2.0 )
        meansqdist_arr[i][j] = np.sum( (DlEE_mean-DlEE_theory_binned)**2.0 + (DlBB_mean-DlBB_theory_binned)**2.0 )

# find the optimal theta_max and apod_sigma values
# chi2 metric
chi2_opt = np.amin(chi2_arr)
chi2_opt_inds = np.unravel_index(np.argmin(chi2_arr), chi2_arr.shape)
print "optimal values by chi2 metric:"
print "theta_max = ",theta_max_arr[chi2_opt_inds[0]]
print "apod_sigma = ",apod_sigma_arr[chi2_opt_inds[0]][chi2_opt_inds[1]]
print "chi2 = ",chi2_opt
np.savetxt(mask_dir+mask_name+"_chi2"+TXT_end, chi2_arr)
# mean-sq-dist metric
meansqdist_opt = np.amin(meansqdist_arr)
meansqdist_opt_inds = np.unravel_index(np.argmin(meansqdist_arr), meansqdist_arr.shape)
print "optimal values by meansqdist metric:"
print "theta_max = ",theta_max_arr[meansqdist_opt_inds[0]]
print "apod_sigma = ",apod_sigma_arr[meansqdist_opt_inds[0]][meansqdist_opt_inds[1]]
print "meansqdist = ",meansqdist_opt
np.savetxt(mask_dir+mask_name+"_meansqdist"+TXT_end, meansqdist_arr)
