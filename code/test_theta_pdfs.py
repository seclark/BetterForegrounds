from __future__ import division, print_function
import numpy as np
import healpy as hp
from bayesian_machinery import *
from plot_map_outcomes import nonzero_data
from astropy.io import fits

import rht_to_planck

sys.path.insert(0, '../../FITSHandling/code/')
import cutouts

# this is how i selected a point
#w = cutouts.make_wcs(allsky_fn)
#cutouts.radec_to_xy(230., 28., w)
#output = (7800.5059999879995, 1795.4966400067201)

hp0orig = fits.getdata('/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_0_healpixproj.fits')
hp0new = fits.getdata('/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_0_healpixproj_nanmask.fits')

#y_pos = 1795
#x_pos = 7800
#y_pos = 1795
#x_pos = 17800

# np.where(hppos == testids[31]) this corresponds to id_ring=18691216
y_pos = 112
x_pos = 5171

stb_root = '/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/single_theta_backprojections/'
galfa_fn = stb_root + 'GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_0.fits'
gg = fits.getdata(galfa_fn)
galfa_hdr = fits.getheader(galfa_fn)

rht_data_unproj = np.zeros(165)
rht_data_proj = np.zeros(165)
for _i in np.arange(165):
    singlethet_fn = stb_root + 'GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_' + str(_i) +'.fits'
    singlethet_data = fits.getdata(singlethet_fn)
    rht_data_unproj[_i] = singlethet_data[y_pos, x_pos]


thet0data = fits.getdata(stb_root + 'GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_0.fits')
allsky_zeros = np.zeros(gg.shape)
allsky_zeros[:, :] = None
allsky_zeros[y_pos, x_pos] = thet0data[y_pos, x_pos]#1

allsky_zeros_block = copy.copy(allsky_zeros)
radblock = 2
allsky_zeros_block[y_pos-radblock:y_pos+radblock, x_pos-radblock:x_pos+radblock] = thet0data[y_pos-radblock:y_pos+radblock, x_pos-radblock:x_pos+radblock]


#allsky_zeros_hp_Gal = rht_to_planck.interpolate_data_to_hp_galactic(allsky_zeros, galfa_hdr)
allsky_zeros_hp_Gal_none, allsky_zeros_hp_Gal_hdr = rht_to_planck.interpolate_data_to_hp_galactic(allsky_zeros, galfa_hdr, nonedata=None)

#allsky_zeros_block_hp_Gal_none, allsky_zeros_block_hp_Gal_hdr = rht_to_planck.interpolate_data_to_hp_galactic(allsky_zeros_block, galfa_hdr, nonedata=None)


# transforming this mask gives me a nest hp index of 3352106 for the nonzero value.

# rht data
rht_cursor, tablename = get_rht_cursor(region="allsky")
#allids = get_all_rht_ids(rht_cursor, tablename)

id = np.int(np.where(allsky_zeros_hp_Gal_none > 0)[0])

"""
for _i in np.arange(165):
    singlethet_fn = stb_root + 'GALFA_HI_allsky_-10_10_w75_s15_t70_thetabin_' + str(_i) +'.fits'
    singlethet_data = fits.getdata(singlethet_fn)

    allsky_zeros_block = np.zeros(gg.shape)
    allsky_zeros_block[:, :] = None
    allsky_zeros_block[y_pos-radblock:y_pos+radblock, x_pos-radblock:x_pos+radblock] = singlethet_data[y_pos-radblock:y_pos+radblock, x_pos-radblock:x_pos+radblock]
    allsky_zeros_block_hp_Gal_none, allsky_zeros_block_hp_Gal_hdr = rht_to_planck.interpolate_data_to_hp_galactic(allsky_zeros_block, galfa_hdr, nonedata=None)
    rht_data_proj[_i] = allsky_zeros_block_hp_Gal_none[id]
    print("theta {} done".format(_i))
"""    
    
# to find overlap between allids and single pixels in the interpolation to hp Gal:
# numpix = hp.fitsfunc.read_map("/Volumes/DataDavy/Foregrounds/coords/data_count_hp_projection_numpix.fits")
# onepix = np.copy(numpix)
# onepix[np.where(onepix > 1)] = 0
# zz=np.nonzero(onepix)
# allids = get_all_rht_ids(rht_cursor, tablename)
# oneid = [val for val in zz[0] if val in allids]


#id = 3352106
pp = Posterior(id, rht_cursor=rht_cursor, adaptivep0=False)

thets_unproj = rht_to_planck.get_thets(75)

plt.figure()
plt.plot(thets_unproj, rht_data_unproj)
plt.plot(pp.prior_obj.sample_psi0, pp.prior_obj.rht_data)
plt.legend(['unprojected data', 'projected data'])
plt.xlabel('angle')

print(hp0orig[id], hp0new[id], allsky_zeros_hp_Gal_none[id], thet0data[y_pos, x_pos], allsky_zeros_block_hp_Gal_none[id])
print(pp.prior_obj.unrolled_rht_data[0], pp.prior_obj.rht_data[0], rht_data_unproj[0])