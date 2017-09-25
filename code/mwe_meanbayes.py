from __future__ import division, print_function
import numpy as np

sample_x = np.linspace(0, 1, 100)
sample_y = np.linspace(0, np.pi, 100)
    
y_dx = sample_y[1] - sample_y[0]
x_dx = sample_x[1] - sample_x[0]
        
yarr = sample_y[:, np.newaxis]
xarr = sample_x

# pick some values
ymeas = 0.01
xmeas = 0.1

fwhm = 0.3

# construct 2D gaussian
gaussian = np.exp(-4*np.log(2) * ((xarr-xmeas)**2 + (yarr-ymeas)**2) / fwhm**2)

integrated_over_y = np.trapz(gaussian, dx = y_dx, axis = 1)
integrated_over_x_and_y = np.trapz(integrated_over_y, dx = x_dx)

normed_gaussian = gaussian/integrated_over_x_and_y

# y moment
aintegrand = np.trapz(normed_gaussian*sample_x, axis=1, dx=x_dx)
a = np.trapz(aintegrand, dx=y_dx)

print("a: expected {}, got {}".format(xmeas, a))

# x moment
bintegrand = np.trapz(normed_gaussian*sample_y[:, np.newaxis], axis=1, dx=x_dx)
b = np.trapz(bintegrand, dx=y_dx)

print("b: expected {}, got {}".format(ymeas, b))
 