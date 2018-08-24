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
# This file is used for evaluating how the input can be well reconstructed as the output.
import numpy as np
import cv2
import scipy.io as sio

import params
import autoencoder.model as ae_model
import autoencoder.inference as ae_infer
import visualizer.drawer as vis


def test_infer_single(in_filename, out_filename_gt, out_filename,
                      list_n_features_encoder=ae_model.default_n_features_encoder,
                      list_layer_type_encoder=ae_model.default_layer_type_encoder,
                      list_n_features_decoder=ae_model.default_n_features_decoder,
                      list_layer_type_decoder=ae_model.default_layer_type_decoder,
                      filename_model=params.FILENAME_DEFAULT_MODEL):


    #########################################################
    # read the input HS file
    #########################################################
    # for mat files
    mat_obj = sio.loadmat(in_filename)
    img_hs = mat_obj['img_hs']
    img_hs = img_hs.astype(np.float32) / 65535.0
    img_hs = np.expand_dims(img_hs, axis=0)
    data = img_hs

    n_samples, h, w, chs = data.shape

    #########################################################
    # infer
    #########################################################
    recon, code = ae_infer.infer_ae(data,
                           list_n_features_encoder=list_n_features_encoder,
                           list_layer_type_encoder=list_layer_type_encoder,
                           list_n_features_decoder=list_n_features_decoder,
                           list_layer_type_decoder=list_layer_type_decoder,
                           filename_model=filename_model)

    #########################################################
    # show
    #########################################################
    for i in range(n_samples):
        gt_i = data[i]
        recon_i = recon[i]
        code_i = code[i]

        # visualize the feature map
        vis.visualize_sparse_code(code, rows=8, cols=8, title='nonlinear representation', scale=0.25)
        cv2.waitKey()
        print('Press any keys to continue')

        # compare gt and recon
        img_h, img_w, img_chs = gt_i.shape
        scale = 5
        recon_i_resize = cv2.resize(recon_i, dsize=(img_w / scale, img_h / scale))
        recon_i_resize = np.expand_dims(recon_i_resize, 0)
        gt_i_resize = cv2.resize(gt_i, dsize=(img_w / scale, img_h / scale))
        gt_i_resize = np.expand_dims(gt_i_resize, 0)
        vis.draw_the_comparison(recon_i_resize, gt_i_resize, title='image inferred comparison',
                                compute_psnr=True)
        cv2.waitKey()
        print('Press any keys to continue')

    #########################################################
    # save
    #########################################################
    wvls2b = range(400, 701, 10)
    sio.savemat(out_filename, {'x_recon': np.squeeze(recon), 'wvls2b': wvls2b})
    sio.savemat(out_filename_gt, {'x_recon': np.squeeze(data), 'wvls2b': wvls2b})


if __name__=='__main__':
   test_infer_single(in_filename='./inputs/synthetic/CAVE/92.mat',
                     out_filename_gt='outputs/AE_inference/92_gt.mat',
                     out_filename='outputs/AE_inference/92_inferred.mat')


