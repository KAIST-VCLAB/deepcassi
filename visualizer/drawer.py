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
import numpy as np
import cv2
import assessment.quality as qual


def normalize_1ch(img):
    if img.dtype == np.uint8:
        is_uint8 = True
        img_float = img.astype(np.float32)/255.0
    elif img.dtype == np.float32:
        is_uint8 = False
        img_float = img

    max_val = np.max(img)
    min_val = np.min(img)

    img_normalized = (img_float - min_val)/(max_val - min_val)

    if is_uint8:
        img_normalized = img_normalized*255
        img_normalized = img_normalized.astype(np.uint8)

    return img_normalized


def imshow_with_zoom(wname='zoom',img=[], scale = 1.0, interpolation=cv2.INTER_CUBIC):
    if img == []:
        return
    img_zoom = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interpolation)
    cv2.imshow(wname, img_zoom)

def visualize_sparse_code(code, rows=16, cols=16, padding=5, title='code', scale=0.25):
    # visualize the feature map
    _, fmap_h, fmap_w, fmap_chs = code.shape
    vis_h = (padding + fmap_h) * rows + padding
    vis_w = (padding + fmap_w) * cols + padding
    img_vis = np.zeros(shape=(vis_h, vis_w), dtype=np.float32)

    for c in range(fmap_chs):
        code_ic = code[:, :, :, c]
        max_val = np.max(code_ic)
        min_val = np.min(code_ic)
        if not (min_val == max_val):
           code_ic = (code_ic - min_val) / (max_val - min_val)

        # set position
        idx_v = c / cols
        idx_h = c % cols
        pos_y = (padding + fmap_h) * idx_v
        pos_x = (padding + fmap_w) * idx_h

        # draw
        img_vis[pos_y:(pos_y + fmap_h), pos_x:(pos_x + fmap_w)] \
            = code_ic
    img_vis = img_vis * 255
    img_vis = img_vis.astype(np.uint8)
    img_vis_cmap = np.zeros(shape=(vis_h, vis_w, 3), dtype=np.uint8)
    cv2.applyColorMap(src=img_vis, colormap=cv2.COLORMAP_JET, dst=img_vis_cmap)
    imshow_with_zoom(wname=title, img=img_vis_cmap, scale=scale, interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('code', img_vis)a
    cv2.waitKey(10)

def draw_the_comparison(img, img_gt=[],
                      rows=4, cols=10, padding_row=35, padding_col=5, title='comp',
                        scale=1.0,
                        compute_psnr=True
                        ):
    # get shape
    shape_img = img.shape

    if img_gt != []:
        shape_gt = img_gt.shape
        # check if they are the same
        if shape_gt != shape_img:
            print('the size of the two HS images are different')
            return

    # determined the size
    _, h, w, chs = img.shape
    if img_gt != []:
        vis_h = (padding_row + h + padding_row + h)*rows + padding_row
    else:
        vis_h = (padding_row + h)*rows + padding_row
    vis_w = (padding_col + w)*cols + padding_col
    img_vis = np.zeros(shape=(vis_h, vis_w), dtype=np.float32)

    # text setup
    font = cv2.FONT_HERSHEY_TRIPLEX

    if img_gt != []:
        max_val_gt = np.max(img_gt)
        max_val = np.max(img)
        ratio = max_val_gt/max_val
    else:
        ratio = 1.0
    ratio = 1.0

    for c in range(chs):
        img_c = img[:, :, :, c]*ratio
        # set position
        idx_v = c / cols
        idx_h = c % cols

        if img_gt != []:
            gt_c = img_gt[:, :, :, c]
            # the position for GT
            pos_gt_y = int(padding_row + (padding_row + h + padding_row + h) * idx_v)
            pos_gt_x = int(padding_col + (padding_col + w) * idx_h)
            # draw gt
            img_vis[pos_gt_y:(pos_gt_y + h), pos_gt_x:(pos_gt_x + w)] \
                = gt_c

        # the position for img

        if img_gt != []:
            pos_img_y = pos_gt_y + h + padding_col
        else:
            pos_img_y = padding_row + (padding_row + h) * idx_v
        pos_img_x = padding_col + (padding_col + w) * idx_h

        # draw img
        img_vis[pos_img_y:(pos_img_y + h), pos_img_x:(pos_img_x + w)] \
            = img_c



        # compute PSNR
        if img_gt != [] and compute_psnr:
            psnr_val = qual.psnr_1ch(img_c, gt_c)
            psnr_text = '%.2f'%(psnr_val)
            pos_font_x = pos_gt_x + 10
            pos_font_y = pos_img_y + h + 20
            cv2.putText(img_vis, psnr_text, fontFace=font,
                        org=(pos_font_x, pos_font_y), fontScale=0.75,
                        color=(1.0, 1.0, 1.0), bottomLeftOrigin=False)


    img_vis = np.power(img_vis, 1/2.2)
    img_vis = img_vis * 255
    img_vis = img_vis.astype(np.uint8)
    img_vis_cmap = np.zeros(shape=(vis_h, vis_w, 3), dtype=np.uint8)
    cv2.applyColorMap(src=img_vis, colormap=cv2.COLORMAP_JET, dst=img_vis_cmap)
    imshow_with_zoom(wname=title, img=img_vis, scale=1, interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('code', img_vis)a
    #cv2.waitKey(100)

