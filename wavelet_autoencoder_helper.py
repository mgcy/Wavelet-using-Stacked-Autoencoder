import numpy as np
from math import log10

def snr(image_a, image_b):
    # calculate mean square error between two images
    var_a = np.var(image_a.astype(float))
    var_b = np.var(image_b.astype(float)-image_a.astype(float))
    snr = 10 * log10(var_a/var_b)   
    return snr

def mse(image_a, image_b):
    # calculate mean square error between two images
    mse = np.sum((image_a.astype(float) - image_b.astype(float)) ** 2)
    mse /= float(image_a.shape[0])
    return mse
