
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



