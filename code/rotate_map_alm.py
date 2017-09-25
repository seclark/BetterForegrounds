import numpy
import healpy
import time
import coord_v_convert


def rotate_map(map,iepoch,oepoch,isys,osys,ns):
    #ns=healpy.npix2nside(map.size)
    alm=healpy.map2alm(map)
    phi,theta,psi=coord_v_convert.py_coordsys2euler_zyz(iepoch,oepoch,isys,osys)
    
    #healpy.sphtfunc.rotate_alm(alm,phi,theta,psi)
    healpy.rotate_alm(alm,phi,theta,psi)
    out_map=healpy.alm2map(alm,nside=ns)
    return out_map