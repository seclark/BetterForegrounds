import numpy as np
import sys
import matplotlib.pyplot as plt
import pyfits
import healpy as hp

# where to save output files
out_root = '../spice/'

# read in files
fileAB = np.loadtxt(sys.argv[1]) #cross
fileAA = np.loadtxt(sys.argv[2]) #auto
fileBB = np.loadtxt(sys.argv[3]) #auto
fskyfile = np.loadtxt(sys.argv[4])
outfile = sys.argv[5]

# define the bins
ellmin=40.0
ellmax=1000.0
Nellbins = 6
#binbounds, binstep = np.linspace(ellmin, ellmax, Nellbins+1, endpoint=True, retstep=True)
binbounds = np.logspace(np.log10(ellmin), np.log10(ellmax), num=Nellbins+1, endpoint=True)
Deltaell = np.zeros(Nellbins)
for i in xrange(1,Nellbins+1):
    Deltaell[i-1] = binbounds[i]-binbounds[i-1]

# read in fsky values
fskymask = fskyfile[0]
fsky2mask = fskyfile[1]

# polspice-computed power spectra
# cross
clAB = np.transpose(fileAB[:,1:7]) #order = TT,EE,BB,TE,TB,EB
# auto
clAA = np.transpose(fileAA[:,1:7]) #order = TT,EE,BB,TE,TB,EB
# auto
clBB = np.transpose(fileBB[:,1:7]) #order = TT,EE,BB,TE,TB,EB

# binning function
def bin_cl(cl,bb):
    ell = np.arange(len(cl))
    # multiply by ell*(ell+1)/2pi before binning
    cl *= (ell*(ell+1.0))/(2.0*np.pi)
    print(ell)
    print(bb)
    bb=[np.int(b) for b in bb]
    ellsubarrs = np.split(ell, bb)
    clsubarrs = np.split(cl, bb)
    ellbinned = np.zeros(len(bb)+1)
    clbinned = np.zeros(len(bb)+1)
    for i in xrange(len(bb)+1):
        ellbinned[i] = np.mean(ellsubarrs[i])
        clbinned[i] = np.mean(clsubarrs[i])
    # cut out the unwanted elements
    return [ellbinned[1:len(ellbinned)-1], clbinned[1:len(ellbinned)-1]]

# bin
clAB_binned = np.zeros((6,Nellbins))
clAA_binned = np.zeros((6,Nellbins))
clBB_binned = np.zeros((6,Nellbins))
for i in xrange(6):
    print(clAB[i])
    print(binbounds)
    [ell_binned, clAB_binned[i]] = bin_cl(clAB[i], binbounds)
    [ell_binned, clAA_binned[i]] = bin_cl(clAA[i], binbounds)
    [ell_binned, clBB_binned[i]] = bin_cl(clBB[i], binbounds)

# error bars -- no sample variance included for now
Delta_cl_AB = np.zeros((6,Nellbins))
for i in xrange(3): #TT,EE,BB
    Delta_cl_AB[i] = np.sqrt((clAA_binned[i]*clBB_binned[i]) / fskymask / (2.0*ell_binned+1.0) / Deltaell)
#TE,TB,EB -- account for "symmetric_cl" from polspice
Delta_cl_AB[3] = 0.5 * np.sqrt( 1.0 / ((2.0*ell_binned+1.0) * Deltaell * fskymask) * (clAA_binned[0]*clBB_binned[1] + clAA_binned[1]*clBB_binned[0]) )
Delta_cl_AB[4] = 0.5 * np.sqrt( 1.0 / ((2.0*ell_binned+1.0) * Deltaell * fskymask) * (clAA_binned[0]*clBB_binned[2] + clAA_binned[2]*clBB_binned[0]) )
Delta_cl_AB[5] = 0.5 * np.sqrt( 1.0 / ((2.0*ell_binned+1.0) * Deltaell * fskymask) * (clAA_binned[1]*clBB_binned[2] + clAA_binned[2]*clBB_binned[1]) )

# chi^2 w.r.t. null
chi2 = np.zeros(6)
for i in xrange(6):
    chi2[i] = np.sum( clAB_binned[i]**2.0 / Delta_cl_AB[i]**2.0 )
    print np.sqrt(chi2[i])

# save binned results + error bars
# we only want the EE and BB power spectra
# columns are ell , EE , EE_err , BB , BB_err
np.savetxt(out_root+outfile+'.txt', np.transpose([ell_binned, clAB_binned[1], Delta_cl_AB[1], clAB_binned[2], Delta_cl_AB[2]]))

# plot
plt.clf()
plt.axhline(y=0.0,color='k', lw=1.0)
EE, = plt.semilogx(ell_binned, clAB_binned[1], 'bo')
BB, = plt.semilogx(ell_binned, clAB_binned[2], 'rs')
#error bars
plt.errorbar(ell_binned, clAB_binned[1], yerr=[Delta_cl_AB[1],Delta_cl_AB[1]], fmt='bo', ecolor='b', elinewidth=2.5, capsize=5, capthick=2)
plt.errorbar(ell_binned, clAB_binned[2], yerr=[Delta_cl_AB[2],Delta_cl_AB[2]], fmt='rs', ecolor='r', elinewidth=2.5, capsize=5, capthick=2)
plt.xlabel(r"$\ell$",fontsize=18)
plt.ylabel(r"$\ell(\ell+1) C_{\ell}^{EE,BB (353 \times 217)}/2\pi \, [{\rm K}^2]$",fontsize=18)
plt.xlim(left=ellmin,right=ellmax)
#plt.ylim( -1.0e-11, 5.0e-11 )
plt.figlegend( (EE,BB), ('EE','BB'), loc='upper right')
plt.grid()
plt.savefig(out_root+outfile+'.png')
