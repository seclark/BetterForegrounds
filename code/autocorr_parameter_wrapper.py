from __future__ import division, print_function

"""
This is a parameter wrapper to facilitate running autocorrelations.
 
"""

class PolParams():
    """
    Store PolSpice parameters
    """
    def __init__(self, apodsigma=7.65, apodtype=0, thetamax=14.0, nlmax=1000, ellmin=40.0, ellmax=1000.0, nbins=6, mapname="", CLAA="", AAOUT="", beamA=0., maskdir="", maskname="", fskyname="", Nside=2048, coords='G', FWHM_apod_arcmin=15., CLAAbinned="", kernelfile="", kernelbool=1):
        # polspice parameters
        self.apodsigma = apodsigma
        self.apodtype = apodtype
        self.thetamax = thetamax
        self.nlmax = nlmax
        
        # binning parameters
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.nbins = nbins
        
        # various filename parameters
        # note that the weights file and fsky file (and their names) are constructed automatically
        self.mapname = mapname
        self.CLAA = CLAA
        self.AAOUT = AAOUT
        self.maskdir = maskdir
        self.maskname = maskname
        self.fskyname = fskyname
        self.Nside= Nside
        self.coords = coords
        self.FWHM_apod_arcmin = FWHM_apod_arcmin #apodization of mask
        self.CLAAbinned = CLAAbinned
        self.kernelfile = kernelfile
        self.kernelbool = kernelbool
        
        # angular resolution of the map (FWHM in arcmin)
        self.beamA = 0.
        

# # Parameter space search code will spit out numbers. Those go here
# params = PolParams(apodsigma=7.65, apodtype=0, thetamax=14.0, nlmax=1000, ellmin=40.0, ellmax=1000.0, nbins=6)

# # filename of code that computes the autocorrelation
# autocorr_routine = 'autocorr_automate'

# # Call routine ofr
# map1fn = "filename"
# weightsA 
# CLAA 
# AAOUT 
# beamA 
# fskyfile 
# CLAAbinned 
# kernelfn 

        
# # call autocorrelation routine, which in turn will call the binning code and output binned autocorrelation spectrum.
# subprocess.call([autocorr_routine, map1fn, params.apodsigma, params.apodtype, params.thetamax, nlmax, weightsA, CLAA, AAOUT, beamA, fskyfile, finalfile, kernelfn, params.ellmin, params.ellmax, params.nbins])        


