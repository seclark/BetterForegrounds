import numpy as np
import sys
import healpy as hp

"""
Code to bin the autocorrelation spectra output
Called from autocorr_automate.sh
"""

# read in files
fileAA = np.loadtxt(sys.argv[1]) #auto
fskyfile = np.loadtxt(sys.argv[2])
outfile = sys.argv[3] # this should include the root

# read in bin parameters
ellmin = float(sys.argv[4])
ellmax = float(sys.argv[5])
Nellbins = int(sys.argv[6])

# define the bins
binbounds = np.logspace(np.log10(float(ellmin)), np.log10(float(ellmax)), num=Nellbins+1, endpoint=True)
binbounds=[np.int(b) for b in binbounds]
Deltaell = np.zeros(Nellbins)
for i in xrange(1,Nellbins+1):
    Deltaell[i-1] = binbounds[i]-binbounds[i-1]

# read in fsky values
fskymask = fskyfile[0]
fsky2mask = fskyfile[1]

# polspice-computed autocorrelation spectra
clAA = np.transpose(fileAA[:,1:7]) #order = TT,EE,BB,TE,TB,EB

# binning function
def bin_cl(cl,bb):
    ell = np.arange(len(cl))
    # multiply by ell*(ell+1)/2pi before binning
    cltemp = cl*(ell*(ell+1.0))/(2.0*np.pi)
    #print(ell)
    #print(bb)
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

# bin
clAA_binned = np.zeros((6,Nellbins))
for i in xrange(6):
    #print("bin bounds:", binbounds)
    [ell_binned, clAA_binned[i]] = bin_cl(clAA[i], binbounds)

# save binned results + error bars
# we only want the EE and BB autocorrelation spectra. No errors.
# columns are ell , EE , BB 
#np.savetxt(outfile+'.txt', np.transpose([ell_binned, clAA_binned[1], clAA_binned[2]]))
np.savetxt(outfile, np.transpose([ell_binned, clAA_binned[1], clAA_binned[2]]))
