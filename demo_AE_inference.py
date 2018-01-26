


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
    for i in xrange(n_samples):
        gt_i = data[i]
        recon_i = recon[i]
        code_i = code[i]

        # visualize the feature map
        vis.visualize_sparse_code(code, rows=8, cols=8, title='nonlinear representation', scale=0.25)
        cv2.waitKey()
        print 'Press any keys to continue'

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
        print 'Press any keys to continue'

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


