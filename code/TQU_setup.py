import numpy as np
import matplotlib.pyplot as plt
import sys
import healpy as hp
"""
code to set up correctly arranged TQU maps for use in polspice scripts
"""
# parameters
Nside=2048
Npix = 12*Nside**2

# read in T_353 map, p_ML map, psi_ML map, output map file
T353file = sys.argv[1]
pMLfile = sys.argv[2]
psiMLfile = sys.argv[3]
outputfile = sys.argv[4]
#
T353 = hp.fitsfunc.read_map(T353file, field=0)
pML = hp.pixelfunc.reorder(hp.fitsfunc.read_map(pMLfile, field=0, nest=True), n2r=True)
psiML = hp.pixelfunc.reorder(hp.fitsfunc.read_map(psiMLfile, field=0, nest=True), n2r=True)
# check map images (RING vs NESTED, coordinate systems, etc.)
#plt.clf()
#hp.mollview(T353, unit='K_CMB', title='353 GHz Full Mission T', min=-1e-4, max=1e-3)
#plt.show()
#plt.clf()
#hp.mollview(pML, unit='no unit', title='353 GHz P (with RHT) / I Full Mission', min=-0.1, max=0.6)
#plt.show()
#plt.clf()
#hp.mollview(psiML, unit='no unit', title='353 GHz psi (with RHT) Full Mission', min=0.0, max=np.pi)
#plt.show()

print('patching with flat-prior planck data where rht prior data is 0')
#planckpsimap=hp.pixelfunc.reorder(hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits", field=0, nest=True), n2r=True)
#planckpmap=hp.pixelfunc.reorder(hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/BayesianMaps/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits", field=0, nest=True), n2r=True)
planckpsimap=hp.pixelfunc.reorder(hp.fitsfunc.read_map("/Users/susanclark/Dropbox/Foregrounds/BayesianMaps/psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits", field=0, nest=True), n2r=True)
planckpmap=hp.pixelfunc.reorder(hp.fitsfunc.read_map("/Users/susanclark/Dropbox/Foregrounds/BayesianMaps/pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits", field=0, nest=True), n2r=True)


wherezero = np.where(psiML == 0)
pML[wherezero] = planckpmap[wherezero]
psiML[wherezero] = planckpsimap[wherezero]

"""
maxrhtcut = 0.25
print('patching with flat-prior planck data where maxrht < {}'.format(maxrhtcut))
maxrht = hp.fitsfunc.read_map('/Volumes/DataDavy/Foregrounds/BayesianMaps/vel_-10_10_maxrht.fits')
wherepatch = np.where(maxrht < maxrhtcut)
whereboth = np.where((maxrht < maxrhtcut) & (psiML == 0))
print('patching with flat-prior planck data for {} pixels'.format(len(whereboth[0])))
pML[wherepatch] = planckpmap[wherepatch]
psiML[wherepatch] = planckpsimap[wherepatch]
"""

# construct polarized intensity P from polarization fraction (make sure T353 map matches that used in the bayesian code...)
P353 = pML*T353
# construct Q and U from P and psi
#print('hack! adding pi/2 to psi')
#psiML = np.mod(psiML + np.pi/2, np.pi)
QML = P353*np.cos(2.0*psiML)
UML = P353*np.sin(2.0*psiML)
# form appropriate TQU file to be used in polspice
template = np.zeros((3,Npix))
template[0] = T353 #just a placeholder, basically
template[1] = QML
template[2] = UML
# save for use in polspice code
hp.fitsfunc.write_map(outputfile, template, coord='G')
