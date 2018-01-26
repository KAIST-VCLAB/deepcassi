
import numpy as np


def mse_1ch(img, ref):

    diff = np.squeeze(img - ref)
    diff_sqr = np.multiply(diff, diff)
    diff_sqr_sum = np.sum(diff_sqr)
    mse = diff_sqr_sum/(diff.shape[0]*diff.shape[1])
    return mse

def psnr_1ch(img, ref):
    mse = mse_1ch(img, ref)
    #print mse
    psnr = 20.0*np.log10(1.0/np.sqrt(mse))
    return psnr