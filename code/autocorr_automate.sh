#! /bin/bash
# for polspice
#HEALPIXDATA=/home/seclark/Healpix_3.31/data
#HEALPIX=/home/seclark/Healpix_3.31
#SPICE=/home/seclark/PolSpice_v03-04-01/src/spice
HEALPIXDATA=/usr/local/Healpix/data
HEALPIX=/usr/local/Healpix
SPICE=/usr/local/bin64/spice

# ******************************************************************************
# This code is for the *autocorrelation* of one masked map.
# Especially useful for an automated calibration of the polspice mask parameters.
#
# CALL SEQUENCE:
# autocorr_automate.sh map1fn apodsigma apodtype thetamax nlmax weightsA CLAA AAOUT beamA fskyfile CLAAbinned kernelfn ellmin ellmax nbins kernelbool
# ******************************************************************************

# if [ $# -gt 0 ]; then
#     echo "You entered $# arguments"
#     echo "mapfn1 is ${1}"
#     echo "apodsigma is ${2}"
#     echo "apodtype is ${3}"
#     echo "thetamax is ${4}"
#     echo "nlmax is ${5}"
#     echo "weightsA is ${6}"
#     echo "CLAA is ${7}"
#     echo "AAOUT is ${8}"
#     echo "beamA is ${9}"
#     echo "fskyfile is ${10}"
#     echo "CLAAbinned is ${11}"
#     echo "kernelfn is ${12}"
#     echo "ellmin is ${13}"
#     echo "ellmax is ${14}"
#     echo "nbins is ${15}"
#     echo "kernelbool is ${16}"
# else
#     echo "WARNING. You did not use any arguments."
# fi

# input map filename
mapA=${1}

# cross-correlate with a different polarized dust tracer (e.g., 217 GHz or a different split of the 353 GHz data)
# polspice parameters -- calibrated for the mask used in CHPP15, need to be re-calibrated if mask changes
# also note use of -symmetric_cl YES in the code below (important if one wants TE, TB, and EB results)
APODSIGMA=$2
APODTYPE=$3
THETAMAX=$4
NLMAX=$5 #maximum multipole

# mask fns
weightsA=${6}

# Define output power spectra files
CLAA=${7}
AAOUT=${8}

# Gaussian FWHM in arcmin for each map
beamA=${9}

# fskyfile which contains two numbers, Sum(mask)/Npix
fskyfile=${10}

# final output filename for binned power spectrum
CLAAbinned=${11}

# kernel filename
kernelfn=${12}

# binning parameters for autocorr_bin.py
ellmin=${13}
ellmax=${14}
nbins=${15}

# boolean for Polspice kernelsfileout parameter
kernelbool=${16}

# auto-correlations
if [ "${kernelbool}" -eq "1" ]; then
    #echo "Computing with kernelsfileout."
    $SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -clfile $CLAA -decouple YES -mapfile $mapA -weightfile $weightsA -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -kernelsfileout $kernelfn -thetamax $THETAMAX > $AAOUT
else
    $SPICE -apodizesigma $APODSIGMA -apodizetype $APODTYPE -beam $beamA -clfile $CLAA -decouple YES -mapfile $mapA -weightfile $weightsA -nlmax $NLMAX -pixelfile YES -polarization YES -subav YES -symmetric_cl YES -kernelsfileout NO -thetamax $THETAMAX > $AAOUT
fi

# bin and save the autocorrelation spectrum.
python autocorr_bin.py $CLAA $fskyfile $CLAAbinned $ellmin $ellmax $nbins
