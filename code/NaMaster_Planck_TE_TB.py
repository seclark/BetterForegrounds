import numpy as np
import matplotlib
from cycler import cycler
matplotlib.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])
matplotlib.use('pdf')
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
fontProperties = {'family':'sans-serif',
    'weight' : 'normal', 'size' : 20}
import matplotlib.pyplot as plt
#import pyfits
import healpy as hp
import subprocess
import os
### janky NaMaster import ###
try:
    import pymaster as nmt
except ImportError:
    #print "no pymaster module found, but we try again and it works hooray"
    import pymaster as nmt
### --------------------- ###
"""
TODO:
- update to use HM splits for 353-auto TE and TB -- note CMB will be present in 353 auto TE (and 353x217)
- run on all data sets, run on different masks

via Susan:
Mask-wise, my perspective is that it is most interesting to show:
* GAL070 mask for TB, TE, for each of 353, 545, 857 (maybe just TB for 353 and/or 217). Takeaway point: real-ness of signal, increased SNR.

* High-latitude (|b| > 30 or 60, or both) masks, split into above/below the Galactic Plane. If we see similar things in 545 and 857 GHz I think it makes sense to just show one w/ higher SNR (857, I'm assuming). Above/below the GP is of interest to people who study large-scale helicity in the Galactic magnetic field. To my knowledge there is no work on TB signatures of that field specifically, but per this conference my understanding is that whether or not TB changes sign across the Plane is an important data point.

* The longitude masks I sent before. Maybe these aren't actually interesting, but they are simple masks and I am still curious to know the answer. 
"""

#####
FITS_end = '.fits'
TXT_end = '.txt'
PDF_end = '.pdf'
DAT_end = '.dat'
#####

#####
# mask parameters
Nside=2048
N_side=2048
Npix = 12*N_side**2
Coords = 'G' #coordinate system
#mask_dir = '/data/jch/Planckdata/'
#mask_name = 'HFI_Mask_GalPlane-apo0_2048_R2.00'
#mask_field = 3 #GAL070
#mask_field_name = 'GAL070'
mask_dir = '/data/jch/Planckdata/TBornotTB/'
mask_name = 'hi_mask_60deg'
mask_field = 0
mask_field_name = ''
# apodization scale
apod_Arcmin = 60.
apod_Deg = apod_Arcmin/60.
# mask file name
mask_file = mask_dir+mask_name+FITS_end
# read in mask and apodize
# to use the pure-B formalism, need the mask to be differentiable at the boundary; the "C1" and "C2" schemes in nmt satisfy this criterion
mask = hp.ud_grade(hp.read_map(mask_file, verbose=False, field=mask_field), nside_out=Nside)
#print float(np.sum(mask))/float(Npix) #check fsky
apod_Type = 'C2'
#print "apodizing mask"
mask_apod = nmt.mask_apodization(mask, apod_Deg, apotype=apod_Type)
#print float(np.sum(mask_apod))/float(Npix) #check apodized fsky
#print "done apodizing"
fsky = np.sum(mask_apod) / float(Npix)
fsky2 = np.sum(mask_apod**2) / float(Npix)
fsky3 = np.sum(mask_apod**3) / float(Npix)
fsky4 = np.sum(mask_apod**4) / float(Npix)
print 'fsky, fsky2, fsky3, fsky4 =',fsky,fsky2,fsky3,fsky4
#####

#####
# specify ell range
ellmax = 1001
ell = np.arange(int(ellmax)+1)
# data directory and maps and beams
# beams
FWHM_217 = 5.02 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
FWHM_353 = 4.94 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
FWHM_545 = 4.83 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
FWHM_857 = 4.64 #arcmin from Table 6 of https://arxiv.org/pdf/1502.01587v2.pdf
# data
data_dir = '/data/jch/Planckdata/'
# 353 GHz E and B
freq1 = 353
data_name1 = 'HFI_SkyMap_353-psb_2048_R3.01_full'
beam1 = (hp.gauss_beam(fwhm=FWHM_353*(np.pi/180.0/60.0), lmax=3*Nside-1, pol=True))[:,1] # extract polarized beam (this is E, should be identical for B)

# T from 217, 353, 545, 857 (or others if desired)
# loop through all at once
N_freq2 = 4
freq2_arr = [217,353,545,857]
data_name2_arr = ['HFI_SkyMap_217_2048_R3.01_full','HFI_SkyMap_353-psb_2048_R3.01_full','HFI_SkyMap_545_2048_R3.01_full','HFI_SkyMap_857_2048_R3.01_full']
FWHM2_arr = [FWHM_217,FWHM_353,FWHM_545,FWHM_857]
#####

#####
# define binning scheme
binwidth = 30
# Cl binning
bins = nmt.NmtBin(N_side, nlb=binwidth, lmax=int(ellmax))
ell_binned = bins.get_effective_ells()
nbins = len(ell_binned)
# Dl binning
ells_temp = np.arange(3 * N_side, dtype='int32')  # Array of multipoles
weights_temp = (1./float(binwidth)) * np.ones_like(ells_temp)  # Array of weights
bpws = -1 + np.zeros_like(ells_temp)  # Array of bandpower indices
bin_weight = []
i = 0
while binwidth * (i + 1) + 2 < 3 * N_side:
    bpws[binwidth * i + 2:binwidth * (i + 1) + 2] = i
    bin_weight.append(np.sum((1./float(binwidth)) * (ells_temp[binwidth * i + 2:binwidth * (i + 1) + 2])*(ells_temp[binwidth * i + 2:binwidth * (i + 1) + 2]+1.)/2./np.pi))
    i += 1
#bins2 = nmt.NmtBin(N_side, bpws=bpws, ells=ells_temp, weights=weights_temp, lmax=int(binwidth * np.floor(ellmax/binwidth))+1) #turns out nlb behavior with ellmax above is non-trivial
# is_Dell and f_ell are not implemented yet in my version
#bins2 = nmt.NmtBin(N_side, bpws=bpws, ells=ells_temp, f_ell=ells_temp*(ells_temp+1.)/2./np.pi, lmax=int(binwidth * np.floor(ellmax/binwidth))+1) #turns out nlb behavior with ellmax above is non-trivial
#bins2 = nmt.NmtBin(N_side, nlb=binwidth, is_Dell=True) #not implemented yet in my version
bins2 = nmt.NmtBin(N_side, bpws=bpws, ells=ells_temp, weights=weights_temp * ells_temp*(ells_temp+1.)/2./np.pi, lmax=int(binwidth * np.floor(ellmax/binwidth))+1)
ell_binned2 = bins2.get_effective_ells()
nbins2 = len(ell_binned2)
#scalefac = ell_binned2*(ell_binned2+1.)/2./np.pi
#scalefac = np.ones(len(ell_binned2))
bin_weight = bin_weight[0:nbins2]
scalefac=bin_weight
#####

#####
# error bars
def error_bar_gauss(Cl12,Cl11,Cl22,fs2,fs4,ellbinned,Deltaell):
    #print (Cl11*Cl22 + Cl12**2.) / (fs2**2./fs4) / (2.*ellbinned+1) / Deltaell
    # annoyingly seems namaster sometimes yields negative values for mask-deconvolved auto-spectra...
    #if any(x < 0 for x in Cl11):
    print Cl11[(np.where(Cl11<0))[0]]
    #if any(x < 0 for x in Cl22):
    print Cl22[(np.where(Cl22<0))[0]]
    return np.sqrt( (Cl11*Cl22 + Cl12**2.) / (fs2**2./fs4) / (2.*ellbinned+1) / Deltaell)
#####

#####
# super naive SNR (chi2 w.r.t. null, so negative points also contribute)
def SNR(Cl12,DeltaCl12):
    return np.sqrt( np.sum(Cl12**2/DeltaCl12**2) )
#####

#####
# read in the maps and compute the mode-coupling matrix for non-pure and pure fields
Q_1, U_1 = hp.read_map(data_dir+data_name1+FITS_end, field=[1, 2], verbose=False)
# non-pure
EB_npure_1 = nmt.NmtField(mask_apod, [Q_1,U_1], beam=beam1)
# pure
EB_pure_1 = nmt.NmtField(mask_apod, [Q_1,U_1], purify_e = True, purify_b = True)

# loop over freq2
for i in range(N_freq2):
    # read in the maps and compute the mode-coupling matrix for non-pure and pure fields
    T_2 = hp.read_map(data_dir+data_name2_arr[i]+FITS_end, field=[0], verbose=False)
    beam2 = (hp.gauss_beam(fwhm=FWHM2_arr[i]*(np.pi/180.0/60.0), lmax=3*Nside-1, pol=False)) # extract beam
    # non-pure
    T_npure_2 = nmt.NmtField(mask_apod, [T_2], beam=beam2)
    w_npure = nmt.NmtWorkspace()
    w_npure.compute_coupling_matrix(T_npure_2, EB_npure_1, bins)
    w_npure_Dl = nmt.NmtWorkspace()
    w_npure_Dl.compute_coupling_matrix(T_npure_2, EB_npure_1, bins2)
    # "compute_full_master" -- gives identical results to decouple_cell below
    Cl_TE_TB = nmt.compute_full_master(T_npure_2, EB_npure_1, bins)
    Cl_TE = Cl_TE_TB[0]
    Cl_TB = Cl_TE_TB[1]
    Dl_TE_TB = nmt.compute_full_master(T_npure_2, EB_npure_1, bins2)
    Dl_TE = Dl_TE_TB[0] * scalefac
    Dl_TB = Dl_TE_TB[1] * scalefac
    # gaussian covariance -- not yet implemented in my version
    #cw = nmt.NmtCovarianceWorkspace()
    #cw.compute_coupling_coefficients(T_npure_2, T_npure_2, T_npure_2, T_npure_2)
    # also compute auto-spectra for error bars
    Cl_TT = nmt.compute_full_master(T_npure_2, T_npure_2, bins)
    Cl_2x2 = nmt.compute_full_master(EB_npure_1, EB_npure_1, bins)
    Cl_EE = Cl_2x2[0]
    Cl_BB = Cl_2x2[3]
    Dl_TT = nmt.compute_full_master(T_npure_2, T_npure_2, bins2) * scalefac
    Dl_2x2 = nmt.compute_full_master(EB_npure_1, EB_npure_1, bins2)
    Dl_EE = Dl_2x2[0] * scalefac
    Dl_BB = Dl_2x2[3] * scalefac
    # pure
    w_pure = nmt.NmtWorkspace()
    w_pure.compute_coupling_matrix(T_npure_2, EB_pure_1, bins)
    w_pure_Dl = nmt.NmtWorkspace()
    w_pure_Dl.compute_coupling_matrix(T_npure_2, EB_pure_1, bins2)
    # "compute_full_master" -- gives identical results to decouple_cell below
    Cl_TE_TB_pure = nmt.compute_full_master(T_npure_2, EB_pure_1, bins)
    Cl_TE_pure = Cl_TE_TB_pure[0]
    Cl_TB_pure = Cl_TE_TB_pure[1]
    Dl_TE_TB_pure = nmt.compute_full_master(T_npure_2, EB_pure_1, bins2)
    Dl_TE_pure = Dl_TE_TB_pure[0] * scalefac
    Dl_TB_pure = Dl_TE_TB_pure[1] * scalefac
    # also compute auto-spectra for error bars (TT done above)
    Cl_2x2_pure = nmt.compute_full_master(EB_pure_1, EB_pure_1, bins)
    Cl_EE_pure = Cl_2x2_pure[0]
    Cl_BB_pure = Cl_2x2_pure[3]
    Dl_2x2_pure = nmt.compute_full_master(EB_pure_1, EB_pure_1, bins2)
    Dl_EE_pure = Dl_2x2_pure[0] * scalefac
    Dl_BB_pure = Dl_2x2_pure[3] * scalefac
    #####
    #####
    # theory predictions with mode-coupling accounted for
    #ClDust_binned_npure = w_npure.decouple_cell(w_npure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
    #ClDust_binned_pure = w_pure.decouple_cell(w_pure.couple_cell(np.array([EEDust,EBDust,BEDust,BBDust])))
    #####
    #####
    ##these are identical to compute_full_master
    ##non-pure
    #ClTETB = w_npure.decouple_cell(nmt.compute_coupled_cell(T_npure_2, EB_npure_1)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    #ClTE = ClTETB[0]
    #ClTB = ClTETB[1]
    ##pure
    #ClTETB_pure = w_pure.decouple_cell(nmt.compute_coupled_cell(T_npure_2, EB_pure_1)) # Compute pseudo-Cls and deconvolve mask mode-coupling matrix to get binned bandpowers
    #ClTE_pure = ClTETB_pure[0]
    #ClTB_pure = ClTETB_pure[1]
    #####
    #####
    # error bars
    DeltaCl_TE = error_bar_gauss(Cl_TE, Cl_TT, Cl_EE, fsky2, fsky4, ell_binned, binwidth)
    DeltaCl_TB = error_bar_gauss(Cl_TB, Cl_TT, Cl_BB, fsky2, fsky4, ell_binned, binwidth)
    DeltaCl_TE_pure = error_bar_gauss(Cl_TE_pure, Cl_TT, Cl_EE_pure, fsky2, fsky4, ell_binned, binwidth)
    DeltaCl_TB_pure = error_bar_gauss(Cl_TB_pure, Cl_TT, Cl_BB_pure, fsky2, fsky4, ell_binned, binwidth)
    DeltaDl_TE = error_bar_gauss(Dl_TE, Dl_TT, Dl_EE, fsky2, fsky4, ell_binned2, binwidth)
    DeltaDl_TB = error_bar_gauss(Dl_TB, Dl_TT, Dl_BB, fsky2, fsky4, ell_binned2, binwidth)
    DeltaDl_TE_pure = error_bar_gauss(Dl_TE_pure, Dl_TT, Dl_EE_pure, fsky2, fsky4, ell_binned2, binwidth)
    DeltaDl_TB_pure = error_bar_gauss(Dl_TB_pure, Dl_TT, Dl_BB_pure, fsky2, fsky4, ell_binned2, binwidth)
    # naive SNR
    SNR_TE = SNR(Cl_TE, DeltaCl_TE[0])
    SNR_TB = SNR(Cl_TB, DeltaCl_TB[0])
    SNR_TE_pure = SNR(Cl_TE_pure, DeltaCl_TE_pure[0])
    SNR_TB_pure = SNR(Cl_TB_pure, DeltaCl_TB_pure[0])
    SNR_TE_Dl = SNR(Dl_TE, DeltaDl_TE[0])
    SNR_TB_Dl = SNR(Dl_TB, DeltaDl_TB[0])
    SNR_TE_pure_Dl = SNR(Dl_TE_pure, DeltaDl_TE_pure[0])
    SNR_TB_pure_Dl = SNR(Dl_TB_pure, DeltaDl_TB_pure[0])
    print '---'+str(freq2_arr[i])+'---'
    print 'SNR (TE,TB,TE_pure,TB_pure):', SNR_TE, SNR_TB, SNR_TE_pure, SNR_TB_pure
    print 'SNR Dl (TE,TB,TE_pure,TB_pure):', SNR_TE_Dl, SNR_TB_Dl, SNR_TE_pure_Dl, SNR_TB_pure_Dl
    #####
    #####
    # # Plot results - non-pure
    # # comment out for now, as non-pure autos are weird
    # plt.clf()
    # plt.title(str(freq2)+r' GHz T $\times$ '+str(freq1)+' GHz E/B (beam-deconvolved; non-pure estimator)')
    # #plt.semilogy(ell_binned, ClTE, 'bo', label='TE')
    # #plt.semilogy(ell_binned, Cl_TE, 'c*', label='TE (CFM cross-check)')
    # #plt.semilogy(ell_binned*1.1, ClTB, 'mo', label='TB')
    # #plt.semilogy(ell_binned*1.1, Cl_TB, 'r*', label='TB (CFM cross-check)')
    # scale_fac = ell_binned*(ell_binned+1)/2./np.pi
    # plt.errorbar(ell_binned, scale_fac * Cl_TE, yerr=[scale_fac * DeltaCl_TE[0], scale_fac * DeltaCl_TE[0]], fmt='b', ecolor='b', elinewidth=1.5, capsize=3, capthick=1, marker='o', label='TE')
    # plt.errorbar(ell_binned+binwidth/4, scale_fac * Cl_TB, yerr=[scale_fac * DeltaCl_TB[0], scale_fac * DeltaCl_TB[0]], fmt='r', ecolor='r', elinewidth=1.5, capsize=3, capthick=1, marker='o', label='TB')
    # plt.xlim(2, int(ellmax))
    # # ranges for log plots
    # #if freq2 == 545:
    # #    plt.ylim(1.e-13, 1.e-6)
    # #elif freq2 == 857:
    # #    plt.ylim(1.e-13, 3.e-6)
    # #else:
    # #    pass
    # #    #plt.ylim()
    # plt.xlabel(r'$\ell$', fontsize=16)
    # plt.ylabel(r'$D_{\ell} \, [{\rm K} \times {\rm MJy/sr}]$', fontsize=16)
    # plt.axhline(y=0, lw=0.75, color='k')
    # plt.grid()
    # plt.legend(loc='upper right', ncol=1, fontsize=9)
    # plt.tight_layout()
    # plt.savefig(data_dir+str(freq2)+'x'+str(freq1)+'_R3.01_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Dl_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
    # np.savetxt(data_dir+str(freq2)+'x'+str(freq1)+'_R3.01_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Cl_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, ClTE, Cl_TE, ClTB, Cl_TB])))
    #####
    #####
    # Plot results - pure - C_ell
    plt.clf()
    plt.title(str(freq2_arr[i])+r' GHz T $\times$ '+str(freq1)+' GHz E/B (beam-deconvolved; pure)')
    #plt.semilogy(ell_binned, ClTE_pure, 'bo', label='TE')
    #plt.semilogy(ell_binned, Cl_TE_pure, 'c*', label='TE (CFM cross-check)')
    #plt.semilogy(ell_binned*1.1, ClTB_pure, 'mo', label='TB')
    #plt.semilogy(ell_binned*1.1, Cl_TB_pure, 'r*', label='TB (CFM cross-check)')
    #scale_fac = ell_binned*(ell_binned+1)/2./np.pi
    #print ell_binned.shape
    #print scale_fac.shape
    #print Cl_TE_pure.shape
    #print (scale_fac * Cl_TE_pure).shape
    #print DeltaCl_TE_pure.shape
    #print (scale_fac * DeltaCl_TE_pure).shape
    # update:log(abs)
    plt.yscale('log')
    plt.errorbar(ell_binned, np.absolute(Cl_TE_pure), yerr=[DeltaCl_TE_pure[0], DeltaCl_TE_pure[0]], fmt='bo', ecolor='b', elinewidth=1.5, capsize=3, capthick=1, label='TE')
    plt.errorbar(ell_binned+binwidth/4, np.absolute(Cl_TB_pure), yerr=[DeltaCl_TB_pure[0], DeltaCl_TB_pure[0]], fmt='rs', ecolor='r', elinewidth=1.5, capsize=3, capthick=1, label='TB')
    plt.xlim(2, int(ellmax))
    #plt.ylim()
    plt.xlabel(r'$\ell$', fontsize=16)
    if (i==0 or i==1):
        plt.ylabel(r'$|C_{\ell}| \, [{\rm K}^2]$', fontsize=16)
    elif (i==2 or i==3):
        plt.ylabel(r'$|C_{\ell}| \, [{\rm K} \times {\rm MJy/sr}]$', fontsize=16)
    plt.axhline(y=0, lw=0.75, color='k')
    plt.grid()
    plt.legend(loc='upper right', ncol=1, fontsize=9)
    plt.tight_layout()
    plt.savefig(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
    #np.savetxt(data_dir+str(freq2)+'x'+str(freq1)+'_R3.01_NMT'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, ClTE_pure, Cl_TE_pure, ClTB_pure, Cl_TB_pure])))
    if (i==0 or i==1):
        np.savetxt(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, Cl_TE_pure, DeltaCl_TE_pure[0], Cl_TB_pure, DeltaCl_TB_pure[0]])), header='ell_bin_center  Cl_TE  sigma_ClTE  Cl_TB  sigma_ClTB  (all in K_CMB^2)')
    elif (i==2 or i==3):
        np.savetxt(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Clpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned, Cl_TE_pure, DeltaCl_TE_pure[0], Cl_TB_pure, DeltaCl_TB_pure[0]])), header='ell_bin_center  Cl_TE  sigma_ClTE  Cl_TB  sigma_ClTB  (all in K_CMB * MJy/sr)')
    #####
    #####
    # Plot results - pure - D_ell
    plt.clf()
    plt.title(str(freq2_arr[i])+r' GHz T $\times$ '+str(freq1)+' GHz E/B (beam-deconvolved; pure)')
    #plt.semilogy(ell_binned, ClTE_pure, 'bo', label='TE')
    #plt.semilogy(ell_binned, Cl_TE_pure, 'c*', label='TE (CFM cross-check)')
    #plt.semilogy(ell_binned*1.1, ClTB_pure, 'mo', label='TB')
    #plt.semilogy(ell_binned*1.1, Cl_TB_pure, 'r*', label='TB (CFM cross-check)')
    #scale_fac = ell_binned*(ell_binned+1)/2./np.pi
    #print ell_binned.shape
    #print scale_fac.shape
    #print Cl_TE_pure.shape
    #print (scale_fac * Cl_TE_pure).shape
    #print DeltaCl_TE_pure.shape
    #print (scale_fac * DeltaCl_TE_pure).shape
    plt.xscale('linear')
    plt.yscale('linear')
    plt.errorbar(ell_binned2, Dl_TE_pure, yerr=[DeltaDl_TE_pure[0], DeltaDl_TE_pure[0]], fmt='bo', ecolor='b', elinewidth=1.5, capsize=3, capthick=1, label='TE')
    plt.errorbar(ell_binned2+binwidth/4, Dl_TB_pure, yerr=[DeltaDl_TB_pure[0], DeltaDl_TB_pure[0]], fmt='rs', ecolor='r', elinewidth=1.5, capsize=3, capthick=1, label='TB')
    # compare to C_ell multiplied by ell_eff factors
    #plt.plot(ell_binned, ell_binned*(ell_binned+1.)/2./np.pi * Cl_TE_pure, 'bo', label='ClTE resc.', alpha=0.5)
    #plt.plot(ell_binned+binwidth/4, ell_binned*(ell_binned+1.)/2./np.pi * Cl_TB_pure, 'rs', label='ClTB resc.', alpha=0.5)
    plt.xlim(2, int(ellmax))
    #plt.ylim()
    # ranges for lin plots
    #if freq2_arr[i] == 545:
    #    plt.ylim(-5.e-7, 1.e-6)
    #elif freq2_arr[i] == 857:
    #    plt.ylim(-3.e-6, 5.e-6)
    #else:
    #    pass
    #    #plt.ylim()
    # ranges for log plots
    #if freq2 == 545:
    #    plt.ylim(1.e-13, 1.e-6)
    #elif freq2 == 857:
    #    plt.ylim(1.e-13, 3.e-6)
    #else:
    #    pass
    #    #plt.ylim()
    plt.xlabel(r'$\ell$', fontsize=16)
    if (i==0 or i==1):
        plt.ylabel(r'$\ell(\ell+1) C_{\ell} / (2\pi) \, [{\rm K}^2]$', fontsize=16)
    elif (i==2 or i==3):
        plt.ylabel(r'$\ell(\ell+1) C_{\ell} / (2\pi) \, [{\rm K} \times {\rm MJy/sr}]$', fontsize=16)
    plt.axhline(y=0, lw=0.75, color='k')
    plt.grid()
    plt.legend(loc='upper right', ncol=1, fontsize=9)
    plt.tight_layout()
    plt.savefig(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Dlpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+PDF_end)
    if (i==0 or i==1):
        np.savetxt(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Dlpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned2, Dl_TE_pure, DeltaDl_TE_pure[0], Dl_TB_pure, DeltaDl_TB_pure[0]])), header='ell_effective  Dl_TE  sigma_DlTE  Dl_TB  sigma_DlTB  (all in K_CMB^2)')
    elif (i==2 or i==3):
        np.savetxt(data_dir+str(freq2_arr[i])+'x'+str(freq1)+'_R3.01_'+mask_name+'_'+mask_field_name+'_NMT_'+apod_Type+'apodArcmin_'+str(apod_Arcmin)+'_binned_Dlpure_lmax'+str(ellmax)+'_binwidth'+str(binwidth)+TXT_end, np.transpose(np.array([ell_binned2, Dl_TE_pure, DeltaDl_TE_pure[0], Dl_TB_pure, DeltaDl_TB_pure[0]])), header='ell_effective  Dl_TE  sigma_DlTE  Dl_TB  sigma_DlTB  (all in K_CMB * MJy/sr)')
