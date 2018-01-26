
import params
import numpy as np
import cv2



shift_ours = [24, 19, 14, 9, 4, 0,
              -4, -7, -11, -14, -17,
              -20, -23, -25, -28, -30,
              -32, -34, -36, -38, -40, -42,
              -43, -45, -46, -48, -49, -50,
              -51, -53, -54, -56, -57]


def generate_random_mask(h=512, w=512, scale=1):
    mask = np.random.randint(0, 2, size=(int(h / scale), int(w / scale))).astype(np.float32)
    if scale != 1.0:
        mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def shift_random_mask(mask, chs=31, shift=0.1):
    h, w = mask.shape

    shifted_mask = np.zeros(shape=(h, w, chs), dtype=mask.dtype)

    for j in xrange(h):
        for i in xrange(w):
            for ch in xrange(chs):
                m = int(round((1 - shift) * i + shift * ch))
                if m > w:
                    m = m - w
                if m < 0:
                    m = m + w

                shifted_mask[j, i, ch] = mask[j, m]

    # for ch in xrange(chs):
    #     vis.imshow_with_zoom(wname='shifted mask', img=shifted_mask[:,:,ch]*255,
    #                          scale=1.0, interpolation=cv2.INTER_NEAREST)
    #     cv2.waitKey()
    return shifted_mask

def shift_random_mask_for_real(mask, chs=31, shift_list=[]):
    h, w = mask.shape
    shifted_mask = np.zeros(shape=(h, w, chs), dtype=mask.dtype)

    # disperse the coded mask
    for ch in xrange(chs):
        # for ch in params.VALID_SPECTRAL_CHS:
        shift_val = shift_list[ch]
        M = np.float32([[1, 0, shift_val], [0, 1, 0]])
        dst = cv2.warpAffine(mask, M, (w, h))
        #img_coded_mask_shifted = np.roll(mask, shift=shift_val, axis=1)
        #cv2.imshow('mask ch:' + str(ch), dst)
        #cv2.waitKey()
        shifted_mask[:, :, ch] = dst

    return shifted_mask


def generate_shifted_mask_cube(mask, chs=31,
                               shift_list=shift_ours,
                               shift_list_y=np.zeros(shape=(1,31)),
                               SINGLE_CASSI=False):
    h, w = mask.shape
    shifted_mask = np.zeros(shape=(h, w, chs), dtype=mask.dtype)

    # disperse the coded mask
    for ch in xrange(chs):
        # for ch in params.VALID_SPECTRAL_CHS:
        if SINGLE_CASSI:
            img_coded_mask_shifted = mask
        else:
            #shift_val = shift_list[ch]
            #img_coded_mask_shifted = np.roll(mask, shift=shift_val, axis=1)
            shift_val = shift_list[ch]
            shift_val_y = shift_list_y[ch]
            M = np.float32([[1, 0, shift_val], [0, 1, shift_val_y]])
            img_coded_mask_shifted = cv2.warpAffine(mask, M, (w, h))
        shifted_mask[:, :, ch] = img_coded_mask_shifted

    return shifted_mask


def generate_coded_image(img_hs, mask, chs=31, SINGLE_CASSI=False, shift_list=shift_ours):
    img_masked = np.multiply(img_hs, mask)
    if SINGLE_CASSI:
        # do channel-wise shift
        h, w,_ = mask.shape
        img_masked_shifted = np.zeros(shape=(h, w, chs), dtype=mask.dtype)
        for ch in xrange(chs):
            shift_val = shift_list[ch]
            img_shifted = np.roll(img_masked[:,:,ch], shift=shift_val, axis=1)
            img_masked_shifted[:, :, ch] = img_shifted
        img_masked = img_masked_shifted

    img_prj = np.sum(img_masked, 2)
    img_prj_norm = img_prj / float(chs)
    # normalize
    # img_prj_norm = vis.normalize_1ch(img_prj)
    # vis.imshow_with_zoom('SSCSI', img=img_prj_norm, scale=1.0, interpolation=cv2.INTER_NEAREST)
    # cv2.waitKey()

    return img_prj_norm
