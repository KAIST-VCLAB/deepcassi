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
import scipy.io as sio

import params
import modulation

import autoencoder.model as ae_model
import recon.snapshot.reconstruction as HQHS_recon

def demo_recon(SINGLE_CASSI=False,
               SSCSI=True,
               rho=1e-01,
               sparsity=1e-02,
               lambda_alpha_fidelity=1e-01,
               lr=5.0e-02,
               iters_ADMM=10,
               iters_ADAM=100,
               filename_test='',
               out_filename='',
               out_coded_img_filename='',
               summary_dir=''
               ):
    #########################################################
    # define the shape of the model
    #########################################################
    test_n_features_encoder = ae_model.default_n_features_encoder
    test_layer_type_encoder = ae_model.default_layer_type_encoder
    test_n_features_decoder = ae_model.default_n_features_decoder
    test_layer_type_decoder = ae_model.default_layer_type_decoder
    trained_model_filename = params.FILENAME_DEFAULT_MODEL
    #########################################################
    # read inputs
    #########################################################
    mat_obj = sio.loadmat(file_name=filename_test)
    gt_hs = mat_obj['img_hs']
    gt_hs = gt_hs.astype(np.float32) / 65535.0

    img_h, img_w, img_chs = gt_hs.shape
    mask_2d = modulation.generate_random_mask(h=img_h, w=img_w, scale=1.0)

    # generate coded image
    list_dispersion = np.arange(0, 31)
    if SINGLE_CASSI:
        list_dispersion = np.floor(list_dispersion * 0.5)
        list_dispersion = list_dispersion.astype(dtype=np.int32)
        print list_dispersion

    if SSCSI:
        mask3d = modulation.shift_random_mask(mask_2d, shift=0.1)
    else:
        mask3d = modulation.generate_shifted_mask_cube(mask_2d, shift_list=list_dispersion,
                                                       SINGLE_CASSI=SINGLE_CASSI)

    img_snapshot = modulation.generate_coded_image(gt_hs, mask3d,
                                                   SINGLE_CASSI=SINGLE_CASSI,
                                                   shift_list=list_dispersion)

    img_snapshot = np.squeeze(img_snapshot)
    cv2.imwrite(out_coded_img_filename, img_snapshot * 255 * 3)
    cv2.imshow('coded image', img_snapshot)
    print 'Press any keys to continue!'
    cv2.waitKey()

    #########################################################
    # run reconstructing
    #########################################################
    x_recon, wvls2b = HQHS_recon.recon_snapshot(img_snapshot=img_snapshot,
                                                img_mask=mask_2d,
                                                gt_hs=gt_hs,
                                                out_filename=out_filename,
                                                img_n_chs=31,
                                                list_shift=list_dispersion,
                                                SINGLE_CASSI=SINGLE_CASSI,
                                                SSCSI=SSCSI,
                                                param_rho=rho,
                                                param_sparsity=sparsity,
                                                param_lambda_alpha_fidelity=lambda_alpha_fidelity,
                                                param_learning_rate=lr,
                                                n_iters_ADMM=iters_ADMM,
                                                n_iters_ADAM=iters_ADAM,
                                                ENABLE_ALPHA_FIDELITY=True,
                                                list_n_features_encoder=test_n_features_encoder,
                                                list_layer_type_encoder=test_layer_type_encoder,
                                                list_n_features_decoder=test_n_features_decoder,
                                                list_layer_type_decoder=test_layer_type_decoder,
                                                filename_model=trained_model_filename,
                                                do_summarize=True,
                                                summary_dir=summary_dir)
    print x_recon.shape
    print wvls2b

    cv2.waitKey()


def demo_recon_synthetic_CAVE():
    #########################################################
    # Select Modulation Type
    #########################################################
    SINGLE_CASSI = False
    SSCSI = True

    #########################################################
    # params for CAVE dataset
    #########################################################
    rho = 1e-01
    sparsity = 0.01
    lambda_alpha_fidelity = 1e-01
    lr = 5.0e-02
    iters_ADMM = 10
    iters_ADAM = 100
    filename_test = './inputs/synthetic/CAVE/103.mat'
    out_filename = './outputs/recon_synthetic/103_recon.mat'
    out_coded_img_filename = './outputs/recon_synthetic/103_coded_img.png'
    summary_dir = './outputs/recon_synthetic/103_recon_summary'


    #########################################################
    # Do recon
    #########################################################
    demo_recon(SINGLE_CASSI=SINGLE_CASSI,
               SSCSI=SSCSI,
               rho=rho,
               sparsity=sparsity,
               lambda_alpha_fidelity=lambda_alpha_fidelity,
               lr=lr,
               iters_ADMM=iters_ADMM,
               iters_ADAM=iters_ADAM,
               filename_test=filename_test,
               out_filename=out_filename,
               out_coded_img_filename=out_coded_img_filename,
               summary_dir=summary_dir)

def demo_recon_synthetic_KAIST():
    #########################################################
    # Select Modulation Type
    #########################################################
    SINGLE_CASSI = False
    SSCSI = True

    #########################################################
    # params for KAIST dataset
    #########################################################
    rho = 7.5e-02
    sparsity = 1e-02
    lambda_alpha_fidelity = 1e-01
    lr = 5.0e-02
    iters_ADMM = 20
    iters_ADAM = 200
    filename_test = './inputs/synthetic/KAIST/scene03.mat'
    out_filename = './outputs/recon_synthetic/scene03_recon.mat'
    out_coded_img_filename = './outputs/recon_synthetic/scene03_coded_img.png'
    summary_dir = './outputs/recon_synthetic/scene03_recon_summary'

    #########################################################
    # Do recon
    #########################################################
    demo_recon(SINGLE_CASSI=SINGLE_CASSI,
               SSCSI=SSCSI,
               rho=rho,
               sparsity=sparsity,
               lambda_alpha_fidelity=lambda_alpha_fidelity,
               lr=lr,
               iters_ADMM=iters_ADMM,
               iters_ADAM=iters_ADAM,
               filename_test=filename_test,
               out_filename=out_filename,
               out_coded_img_filename=out_coded_img_filename,
               summary_dir=summary_dir)


if __name__=='__main__':
   #demo_recon_synthetic_CAVE()
   demo_recon_synthetic_KAIST()



