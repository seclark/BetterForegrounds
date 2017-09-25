from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits


# load latest attempt
inpath = "/Users/susanclark/BetterForegrounds/data/"
pMLmap = inpath+"pMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"
psiMLmap = inpath+"psiMB_DR2_SC_241_353GHz_adaptivep0_True_new.fits"


