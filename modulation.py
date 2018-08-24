"""
=======================================================================
General Information
-------------------
Codename: DeepCASSI (ACM SIGGRAPH Asia 2017)
Writers: Inchang Choi (inchangchoi@vclab.kaist.ac.kr), Daniel S. Jeon (sjjeon@vclab.kaist.ac.kr), Giljoo Nam (gjnam@vclab.kaist.ac.kr), Min H. Kim (minhkim@vclab.kaist.ac.kr)

Institute: KAIST Visual Computing Laboratory
For information please see the paper:
High-Quality Hyperspectral Reconstruction Using a Spectral Prior ACM SIGGRAPH ASIA 2017, Inchang Choi, Daniel S. Jeon, Giljoo Nam, Diego Gutierrez, Min H. Kim Visit our project http://vclab.kaist.ac.kr/siggraphasia2017p1/ for the hyperspectral image dataset.
Please cite this paper if you use this code in an academic publication.

Bibtex: @Article{DeepCASSI:SIGA:2017,
author = {Inchang Choi and Daniel S. Jeon and Giljoo Nam 
and Diego Gutierrez and Min H. Kim},
title = {High-Quality Hyperspectral Reconstruction 
Using a Spectral Prior},
journal = {ACM Transactions on Graphics (Proc. SIGGRAPH Asia 2017)},
year = {2017},
volume = {36},
number = {6},
pages = {218:1-13},
doi = "10.1145/3130800.3130810",
url = "http://dx.doi.org/10.1145/3130800.3130810",
}
==========================================================================
License Information
-------------------
Inchang Choi, Daniel S. Jeon, Giljoo Nam, Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:

Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
License: GNU General Public License Usage Alternatively, this file may be used under the terms of the GNU General Public License version 3.0 as published by the Free Software Foundation and appearing in the file LICENSE.GPL included in the packaging of this file. Please review the following information to ensure the GNU General Public License version 3.0 requirements will be met: http://www.gnu.org/copyleft/gpl.html.

Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
=======================================================================
"""
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

    for j in range(h):
        for i in range(w):
            for ch in range(chs):
                m = int(round((1 - shift) * i + shift * ch))
                if m > w:
                    m = m - w
                if m < 0:
                    m = m + w

                shifted_mask[j, i, ch] = mask[j, m]

    # for ch in range(chs):
    #     vis.imshow_with_zoom(wname='shifted mask', img=shifted_mask[:,:,ch]*255,
    #                          scale=1.0, interpolation=cv2.INTER_NEAREST)
    #     cv2.waitKey()
    return shifted_mask

def shift_random_mask_for_real(mask, chs=31, shift_list=[]):
    h, w = mask.shape
    shifted_mask = np.zeros(shape=(h, w, chs), dtype=mask.dtype)

    # disperse the coded mask
    for ch in range(chs):
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
    for ch in range(chs):
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
        for ch in range(chs):
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
