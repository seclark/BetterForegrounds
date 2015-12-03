from __future__ import division, print_function
import numpy as np

"""
 Simple psi, p debiasing. 
"""

def asymptotic_estimator(p, sigp, floor = False):

    """
    AS or canonical p estimator.
    Defined in Planck Int XIX Eq. B1 as p_db^2 = p^2 - sigp^2 :: floor = False
    Defined in Montier+ 2014 Eq. 13 as the above if p > sigp, 0 otherwise. :: floor = True
    """    
    
    # Simplest case
    p_db = np.sqrt(p**2 - sigp**2)
