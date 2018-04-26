from __future__ import division, print_function

"""
This is a parameter wrapper to facilitate running autocorrelations.
 
"""

class PolParams():
    """
    Store PolSpice parameters
    """
    def __init__(self, apodsigma=7.65, apodtype=0, thetamax=14.0, nlmax=1000, ellmin=40.0, ellmax=1000.0, nbins=6
                       mapfn="", weightsA="", CLAA="", AAOUT="", beamA=0, fskyfile="", CLAAbinned="", kernelfn=""
                       kernelbool=1):
        # apodization parameters
        self.apodsigma = apodsigma
        self.apodtype = apodtype
        self.thetamax = thetamax
        self.nlmax = nlmax
        
        # binning parameters
        self.ellmin = ellmin
        self.ellmax = ellmax
        self.nbins = nbins
        
        # various filename parameters
        self.mapfn = mapfn
        self.weightsA = weightsA
        self.CLAA = CLAA
        self.AAOUT = AAOUT
        self.fskyfile = fskyfile
        self.CLAAbinned = CLAAbinned
        self.kernelfn = kernelfn
        
        # For real data this would be the native resolution 
        self.beamA = 0
        
        
        
# Parameter space search code will spit out numbers. Those go here
params = PolParams(apodsigma=7.65, apodtype=0, thetamax=14.0, nlmax=1000, ellmin=40.0, ellmax=1000.0, nbins=6)

# filename of code that computes the autocorrelation
autocorr_routine = 'autocorr_automate'

# Call routine ofr
map1fn = "filename"
weightsA 
CLAA 
AAOUT 
beamA 
fskyfile 
CLAAbinned 
kernelfn 

        
# call autocorrelation routine, which in turn will call the binning code and output binned autocorrelation spectrum.
subprocess.call([autocorr_routine, map1fn, params.apodsigma, params.apodtype, params.thetamax, nlmax, weightsA, CLAA, AAOUT, beamA, fskyfile, finalfile, kernelfn, params.ellmin, params.ellmax, params.nbins])        


