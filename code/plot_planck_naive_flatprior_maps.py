from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from matplotlib import rc
import matplotlib.cm as cm
rc('text', usetex=True)

maproot = "/Volumes/DataDavy/Foregrounds/BayesianMaps/"

#p_naive=hp.fitsfunc.read_map(maproot+"pMB_SC_241_353GHz_naive_abspsidx.fits")
#psi_naivep=hp.fitsfunc.read_map(maproot+"psiMB_SC_241_353GHz_naive_abspsidx.fits")

# original planck data
planck353_fn = "/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.02_full_RING.fits"
planckT, planckQ, planckU = hp.fitsfunc.read_map(planck353_fn, field=(0,1,2))
pnaive = np.sqrt(planckQ**2 + planckU**2)/planckT
#psinaive = np.mod(0.5*np.arctan2(planckU, planckQ), np.pi)

pMLmap=hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_trueallsky_353GHz_adaptivep0_True.fits")
psiMLmap=hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_trueallsky_353GHz_adaptivep0_True.fits")
#psiMLmap=hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits")

fig = plt.figure(figsize=(5, 6.5))
clip_cmap = cm.magma
clip_cmap.set_under("w")
hp.mollview(pnaive, sub=211, cmap=clip_cmap, min=0, max=1, cbar=False, title=r"$p_{naive}$")
hp.mollview(pMLmap, sub=212, cmap=clip_cmap, min=0, max=1, title=r"$p_{MB}$")

#for i in maxrhtover2:
#    pp = Posterior(i, rht_cursor=rht_cursor, adaptivep0=True)
#    plot_all_bayesian_components_from_posterior(pp)
#    pylab.savefig("../figures/all_bayesian_components_id_nest_{}.png".format(i)) 
#    plt.close('all')
