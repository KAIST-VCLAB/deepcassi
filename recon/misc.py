
import numpy as np


def np_del_operator(xk):
    batchsize, height, width, n_chs = xk.shape
    G = np.zeros(shape=(batchsize, height, width, n_chs, 2),
                 dtype=xk.dtype)

    # y gradient
    G[:,:-1,:,:,0] -= xk[:,:-1,:,:]
    G[:,:-1,:,:,0] += xk[:,1:,:,:]

    # x gradient
    G[:,:,:-1,:,1] -= xk[:,:,:-1,:]
    G[:,:,:-1,:,1] += xk[:,:,1:,:]

    G = G[:, :-1, :-1, :, :]
    return G

def soft_threshold(v, l, r):
    threshold_val = l/r
    print 'before: '
    print threshold_val
    print np.max(v)
    print np.min(v)
    # print v.shape
    vshape = v.shape
    v = v.flatten()
    v1 = np.copy(v)
    v2 = np.copy(v)
    v3 = np.copy(v)
    # print v.shape

    abs_v = np.abs(v)
    v1[v1 > threshold_val] -= threshold_val
    v2[abs_v < threshold_val] = 0
    v3[v3 < -threshold_val] += threshold_val

    v[v > threshold_val] = v1[v > threshold_val]
    v[abs_v < threshold_val] = 0
    v[v < -threshold_val] = v3[v < -threshold_val]

    v = np.reshape(v, newshape=vshape)

    #print threshold_val
    print 'after: '
    print np.max(v)
    print np.min(v)
    return v