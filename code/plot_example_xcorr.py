from __future__ import division, print_function
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import pyfits
import healpy as hp
import pylab
from matplotlib import rc
rc('text', usetex=True)

class XcorrData():
    def __init__(self, fn):
        self.fn = fn
        self.data = np.loadtxt(self.fn)
        
        # columns are ell , EE , EE_err , BB , BB_err
        self.ell = self.data[:, 0]
        self.EE = self.data[:, 1]
        self.EE_err = self.data[:, 2]
        self.BB = self.data[:, 3]
        self.BB_err = self.data[:, 4]
        
    
# read in files
fn_prefix = "cl_353full_pMB_psiMB_"
fn_suffix = "_217full_pspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_gpixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned.txt"
fn_suffix2 = "_217full_pspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_gpixcorr_UDOWN_bgt30cut_starmask_HFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned_ppatch_rht0.25.txt"

#testname="SC-4_3_g3"
testname="max10"#"r1mmaxpat"#"rht1mmax"#"thetsig30"#"baseamp1"#"thetrht1"#"b0-10_10"#"smoothb0"#"bamp0"#"bamp1E-2"#"baseamp1"#"revrht"#"fixpsihack"#"fixpsi02"#"mynewMB"#"rhtMAP"#"rhttol13"#"rhttol0"#"adaptivep0"
planckdataname = "newplanck"#"pnewMB"#"pMAP"#"ptol13"#"ptol0"#"padaptp02" #"padaptp0" #"p353s2"
AvMname = "thetrht1"#"avmfixw"
testname2="b0-10_10"#"med10ad"#"rhtmedvar"
testname3="rhtmaxvar"

RHT_data_fn = "../spice/"+fn_prefix+testname+fn_suffix
Planck_data_fn = "../spice/"+fn_prefix+planckdataname+fn_suffix
RHT_data_fn_AvM = "../spice/"+fn_prefix+AvMname+fn_suffix
RHT_data_fn_2 = "../spice/"+fn_prefix+testname2+fn_suffix
RHT_data_fn_3 = "../spice/"+fn_prefix+testname3+fn_suffix
#RHT_data_fn_patch = "../spice/"+fn_prefix+"max10ad"+fn_suffix
#RHT_data_fn_patch = "../spice/"+fn_prefix+"med10ad"+fn_suffix2
#RHT_data_fn_AvM = "cl_353full_pMB_psiMB_AvM_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned.txt"

RawPlanck_fn = "cl_353full_217full_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0_EEBB_binned.txt"

#autopower = "cl_353full_pMB_psiMB_${testname}_auto_polspice_RHT_mask_Equ_ch16_to_24_w75_s15_t70_galfapixcorr_UPSIDEDOWN_plusbgt30cut_plusstarmask_plusHFI_Mask_PointSrc_2048_R2.00_TempPol_allfreqs_RING_apodFWHM15arcmin_APODSIG7p65_APODTYPE0_THETAMAX14p0.txt"


RHTdata = XcorrData(RHT_data_fn)
Planckdata = XcorrData(Planck_data_fn)
RHTdataAvM = XcorrData(RHT_data_fn_AvM)
RawPlanckdata = XcorrData(RawPlanck_fn)
RHTdata2 = XcorrData(RHT_data_fn_2)
#RHTdata3 = XcorrData(RHT_data_fn_3)
#RHTdatapatch = XcorrData(RHT_data_fn_patch)

fig = plt.figure(figsize = (10, 8), facecolor = "white")
ax = fig.add_subplot(111)

#ax.semilogx(RHTdata.ell, RHTdata.EE, 'o')
#ax.semilogx(RHTdata.ell, RHTdata.BB, 's')
ellnudge = 0.05

p1 = ax.errorbar(Planckdata.ell*(1 - 2*ellnudge), Planckdata.EE, yerr=[Planckdata.EE_err, Planckdata.EE_err], ms = 7, fmt='o', markerfacecolor='darkblue', mec='white',  ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $\mathrm{EE}$")
p2 = ax.errorbar(RawPlanckdata.ell*(1 + -1*ellnudge), RawPlanckdata.EE, yerr=[RawPlanckdata.EE_err, RawPlanckdata.EE_err], ms = 7, fmt='o', markerfacecolor='lightsteelblue', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$\mathrm{Raw}$ $Planck$ $\mathrm{EE}$")
p3 = ax.errorbar(RHTdata.ell*(1 + 0*ellnudge), RHTdata.EE, yerr=[RHTdata.EE_err, RHTdata.EE_err], ms = 7, fmt='o', markerfacecolor='deepskyblue', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$\mathrm{Offset}$ $R(\psi)$ $\mathrm{prior}$ $\mathrm{EE}$")
#ax.errorbar(RHTdata.ell*(1 + 2*ellnudge), RHTdata.EE, yerr=[RHTdata.EE_err, RHTdata.EE_err], ms = 7, fmt='o', markerfacecolor='cornflowerblue', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $+$ $\mathrm{1 - RHT}$ $\mathrm{EE}$")
p4 = ax.errorbar(RHTdata2.ell*(1 + 1*ellnudge), RHTdata2.EE, yerr=[RHTdata2.EE_err, RHTdata2.EE_err], ms = 7, fmt='o', markerfacecolor='darkturquoise', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$R(\psi)$ $\mathrm{prior}$ $\mathrm{EE}$")
p5 = ax.errorbar(RHTdataAvM.ell*(1 + 2*ellnudge), RHTdataAvM.EE, yerr=[RHTdataAvM.EE_err, RHTdataAvM.EE_err], ms = 7, fmt='o', markerfacecolor='steelblue', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$$+$$\theta_{\mathrm{RHT}}$ $\mathrm{EE}$")
#ax.errorbar(RHTdatapatch.ell*(1 + 3*ellnudge), RHTdatapatch.EE, yerr=[RHTdatapatch.EE_err, RHTdatapatch.EE_err], ms = 7, fmt='o', markerfacecolor='cornflowerblue', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $+$ $\mathrm{1 - RHT}$ $\mathrm{EE}$")

p6 = ax.errorbar(Planckdata.ell*(1 - 2*ellnudge), Planckdata.BB, yerr=[Planckdata.BB_err, Planckdata.BB_err], ms = 7, fmt='s', markerfacecolor='crimson', mec='white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $\mathrm{BB}$")
p7 = ax.errorbar(RawPlanckdata.ell*(1 + -1*ellnudge), RawPlanckdata.BB, yerr=[RawPlanckdata.BB_err, RawPlanckdata.BB_err], ms = 7, fmt='s', markerfacecolor='navajowhite', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$\mathrm{Raw}$ $Planck$ $\mathrm{BB}$")
p8 = ax.errorbar(RHTdata.ell*(1 + 0*ellnudge), RHTdata.BB, yerr=[RHTdata.BB_err, RHTdata.BB_err], ms = 7, fmt='s', markerfacecolor='darkorange', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$\mathrm{Offset}$ $R(\psi)$ $\mathrm{prior}$ $\mathrm{BB}$")
#ax.errorbar(RHTdata.ell*(1 + 2*ellnudge), RHTdata.BB, yerr=[RHTdata.BB_err, RHTdata.BB_err], ms = 7, fmt='s', markerfacecolor='lightcoral', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $+$ $\mathrm{1 - RHT}$ $\mathrm{BB}$")
p9 = ax.errorbar(RHTdata2.ell*(1 + 1*ellnudge), RHTdata2.BB, yerr=[RHTdata2.BB_err, RHTdata2.BB_err], ms = 7, fmt='s', markerfacecolor='orangered', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$R(\psi)$ $\mathrm{prior}$ $\mathrm{BB}$")
p10 = ax.errorbar(RHTdataAvM.ell*(1 + 2*ellnudge), RHTdataAvM.BB, yerr=[RHTdataAvM.BB_err, RHTdataAvM.BB_err], ms = 7, fmt='s', markerfacecolor='lightcoral', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$$+$$\theta_{\mathrm{RHT}}$ $\mathrm{BB}$")
#ax.errorbar(RHTdatapatch.ell*(1 + 3*ellnudge), RHTdatapatch.BB, yerr=[RHTdatapatch.BB_err, RHTdatapatch.BB_err], ms = 7, fmt='s', markerfacecolor='lightcoral', mec = 'white', ecolor = "gray", elinewidth=2, capsize=0, capthick=2, label=r"$Planck$ $+$ $\mathrm{1 - RHT}$ $\mathrm{BB}$")


#print("RHT BB err", RHTdata.BB_err)
#print("Planck BB err", Planckdata.BB_err)

ax.tick_params(labelsize=15)

ax.semilogx()
ax.grid(color="dimgray")
ax.axhline(y=0.0,color='dimgray', lw=1.0)


box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legenditems = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
labelitems = [p.get_label() for p in legenditems]
legendappear = [p1, p2, p3, p4, p5, matplotlib.lines.Line2D([],[],linestyle=''), p6, p7, p8, p9, p10]
labelappear = [r"$\mathrm{Flat}$ $\mathrm{prior}$ $\mathrm{EE}$", r"$\mathrm{Raw}$ $Planck$ $\mathrm{EE}$", r"$\mathrm{Offset}$ $R(\psi)$ $\mathrm{prior}$ $\mathrm{EE}$", 
               r"$R(\psi)$ $\mathrm{prior}$ $\mathrm{EE}$", r"$\psi_{\mathrm{RHT}}$ $\mathrm{AvM}$ $\mathrm{prior}$ $\mathrm{EE}$", 
               "", 
               r"$\mathrm{Flat}$ $\mathrm{prior}$ $\mathrm{BB}$", r"$\mathrm{Raw}$ $Planck$ $\mathrm{BB}$", r"$\mathrm{Offset}$ $R(\psi)$ $\mathrm{prior}$ $\mathrm{BB}$", r"$R(\psi)$ $\mathrm{prior}$ $\mathrm{BB}$", "$\psi_{\mathrm{RHT}}$ $\mathrm{AvM}$ $\mathrm{prior}$ $\mathrm{BB}$"]
ax.legend(legendappear, labelappear, numpoints=1, ncol=1, fancybox=True, borderpad=0.5, bbox_to_anchor=(1.32, 1.0), frameon=False)

#ax.set_ylim( -1.0e-11, 5.0e-11 )
ax.set_xlim(40.0, 1000.0)
ax.set_xlabel(r"$\ell$",fontsize=18)
ax.set_ylabel(r"$\ell(\ell+1) C_{\ell}/2\pi \, [{\rm K}^2]$",fontsize=18)

plt.title(r"$\mathrm{Posterior}$ $\mathrm{Maps}$ $\times$ $217$ $\mathrm{GHz}$", size=20)

# plot
"""
plt.axhline(y=0.0,color='k', lw=1.0)
EE, = plt.semilogx(ell_binned, clAB_binned[1], 'bo')
BB, = plt.semilogx(ell_binned, clAB_binned[2], 'rs')
#error bars
plt.errorbar(ell_binned, clAB_binned[1], yerr=[Delta_cl_AB[1],Delta_cl_AB[1]], fmt='bo', ecolor='b', elinewidth=2.5, capsize=5, capthick=2)
plt.errorbar(ell_binned, clAB_binned[2], yerr=[Delta_cl_AB[2],Delta_cl_AB[2]], fmt='rs', ecolor='r', elinewidth=2.5, capsize=5, capthick=2)
plt.xlabel(r"$\ell$",fontsize=18)
plt.ylabel(r"$\ell(\ell+1) C_{\ell}^{EE,BB (353 \times 217)}/2\pi \, [{\rm K}^2]$",fontsize=18)
plt.xlim(left=ellmin,right=ellmax)
plt.ylim( -1.0e-11, 5.0e-11 )
plt.figlegend( (EE,BB), ('EE','BB'), loc='upper right')
plt.grid()
plt.savefig(outfile+'.png')
"""