import subprocess


"""
This is a shell script to automate running autocorrelations. 
The autocorrelation code is autocorr_automate.sh. It calls Polspice.
Then it calls autocorr_bin.py and outputs binned autocorrelation spectrum.

Call structure for autocorr_automate:
autocorr_automate.sh map1fn apodsigma apodtype thetamax nlmax weightsA CLAA AAOUT beamA fskyfile finalfile kernelfn ellmin ellmax nbins
"""

# call autocorrelation routine, which in turn will call the binning code and output binned autocorrelation spectrum.
subprocess.call([autocorr_routine, map1fn, params.apodsigma, params.apodtype, params.thetamax, nlmax, weightsA, CLAA, AAOUT, beamA, fskyfile, finalfile, kernelfn, params.ellmin, params.ellmax, params.nbins])        
