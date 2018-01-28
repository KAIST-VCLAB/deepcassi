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
import tensorflow as tf
import numpy as np
import params


def build_recon_network(list_n_features_decoder=[],
                        list_layer_type_decoder=[],
                        weight_dict=[],
                        img_h=-1,
                        img_w=-1,
                        img_chs=31,
                        n_features_in_code=64,
                        rho=1.0,
                        learning_rate=1e-01):
    n_convs_decoder = len(list_layer_type_decoder)
    is_reusable = True
    #########################################################
    # Set placeholders
    #########################################################
    # the shape should be (batchsize, psize, psize, n_channels
    # sparse_code_alpha = tf.placeholder(params.TF_DATA_TYPE, name='alpha')
    code_shape = [1, img_h, img_w, n_features_in_code]
    img_gradient_shape = [1, img_h - 1, img_w - 1, img_chs, 2]
    #code_gradient_shape = [1, img_h - 1, img_w - 1, n_features_in_code]

    img_gt = tf.placeholder(params.TF_DATA_TYPE, name='img_gt')
    mask3d = tf.placeholder(params.TF_DATA_TYPE, name='mask3d')

    # xk = tf.Variable(tf.abs(tf.random_normal(shape=code_shape))*0.001, name='xk')
    xk = tf.Variable(tf.ones(shape=code_shape) * 0.0001, name='xk')
    xk_ph = tf.placeholder(params.TF_DATA_TYPE, name='xk_ph')
    op_assign_xk = xk.assign(xk_ph)

    # admm_zk = tf.Variable(tf.abs(tf.random_normal(shape=code_shape))*0.512, name='admm_zk')
    zk = tf.Variable(tf.ones(shape=img_gradient_shape) * 0.0001, name='zk', trainable=False)
    zk_ph = tf.placeholder(params.TF_DATA_TYPE, name='zk_ph')
    op_assign_zk = zk.assign(zk_ph)

    uk = tf.Variable(tf.ones(shape=img_gradient_shape) * 0.0001, name='uk', trainable=False)
    uk_ph = tf.placeholder(params.TF_DATA_TYPE, name='uk_ph')
    op_assign_uk = uk.assign(uk_ph)
    #########################################################
    # Build the decoder
    #########################################################
    layer_name_base = 'decoder'
    response = xk

    for l in xrange(n_convs_decoder):
        layer_name = layer_name_base + '-conv' + str(l)

        list_stride = [1, 1, 1, 1]
        pad = 'SAME'
        with tf.variable_scope(layer_name, reuse=is_reusable):
            key_weight = layer_name + "/weight:0"
            weight_val = weight_dict[key_weight]
            conv_weight = tf.constant(weight_val, name="weight")
            key_bias = layer_name + "/bias:0"
            bias_val = weight_dict[key_bias]
            conv_bias = tf.constant(bias_val, name="bias")
            conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
            response = tf.nn.bias_add(conv, conv_bias)
            response = tf.nn.relu(response)

    # img_recon = response
    img_recon = tf.identity(response, name='img_recon')

    # apply coded mask
    img_masked = tf.mul(img_recon, mask3d, name='masking')
    # projection to 2D
    img_prj = tf.reduce_sum(img_masked, 3)
    # normlaize
    img_prj = img_prj / np.float(img_chs)

    # loss data
    diff = img_prj - img_gt
    loss_data = 1.0 * 1e+00 * 0.5 * tf.reduce_mean(tf.square(diff))

    # loss ADMM
    G_xk_vertical = img_recon[:, 1:, :, :] - img_recon[:, :-1, :, :]
    G_xk_horizontal = img_recon[:, :, 1:, :] - img_recon[:, :, :-1, :]
    G_xk_vertical = G_xk_vertical[:, :, :-1, :]
    G_xk_horizontal = G_xk_horizontal[:, :-1, : :]
    G_xk = tf.pack([G_xk_vertical, G_xk_horizontal], axis=4)
    #G_xk = tf.square(G_xk_vertical) + tf.square(G_xk_horizontal)
    diff = G_xk - zk + uk
    #diff = xk - zk + uk
    loss_admm = 1.0*0.5*rho*tf.reduce_mean(tf.square(diff))

    # regularization on code
    # tv on featrue
    n_total_pixels = img_h * img_w * img_chs
    loss_tv_feature = 0.0 * 1e-02 * (tf.nn.l2_loss(xk[:, 1:, :, :] - xk[:, :img_h - 1, :, :]) / n_total_pixels
                                     + tf.nn.l2_loss(xk[:, :, 1:, :] - xk[:, :, :img_w - 1, :]) / n_total_pixels)

    # tv on recon image
    loss_tv_recon = 0.0 * 1e-02 * (
    tf.nn.l2_loss(img_recon[:, 1:, :, :] - img_recon[:, :img_h - 1, :, :]) / n_total_pixels
    + tf.nn.l2_loss(img_recon[:, :, 1:, :] - img_recon[:, :, :img_w - 1, :]) / n_total_pixels)

    # tv for spectral dimension
    loss_tv_spectrum \
        = 0.0 * 1e-05 * 1.0 * tf.nn.l2_loss(img_recon[:, :, :, 1:] - img_recon[:, :, :, :img_chs - 1]) / n_total_pixels

    # reg
    loss_reg_mag = 0.0 * 1e-06 * 0.5 * tf.reduce_mean((tf.square(xk)))
    loss = loss_data + loss_admm \
           + loss_tv_feature + loss_tv_recon + loss_tv_spectrum \
           + loss_reg_mag
    loss = loss * 1e+00

    # data term
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=1e-01).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # add summary
    loss_summary = tf.scalar_summary('loss', loss)
    summary_op_loss = tf.merge_summary([loss_summary])

    testing_psnr = tf.Variable(0.0, name='var_testing_psnr')
    ph_testing_psnr = tf.placeholder(dtype=tf.float32)
    op_assign_testing_psnr = tf.assign(testing_psnr, ph_testing_psnr)
    testing_psnr_summary = tf.scalar_summary('testing psnr', testing_psnr)
    summary_op_testing_psnr = tf.merge_summary([testing_psnr_summary])

    #########################################################
    # Return model
    #########################################################
    model = {'xk': xk,
             'op_assign_xk': op_assign_xk,
             'xk_ph': xk_ph,
             'zk': zk,
             'op_assign_zk': op_assign_zk,
             'zk_ph': zk_ph,
             'uk': uk,
             'op_assign_uk': op_assign_uk,
             'uk_ph': uk_ph,
             'img_gt': img_gt,
             'mask3d': mask3d,
             'img_recon': img_recon,
             'img_prj': img_prj,
             'loss': loss,
             'loss_data': loss_data,
             'loss_admm': loss_admm,
             'loss_tv_feature': loss_tv_feature,
             'loss_tv_recon': loss_tv_recon,
             'loss_tv_spec': loss_tv_spectrum,
             'loss_reg_mag': loss_reg_mag,
             'optimizer': optimizer,
             'summary_op_loss': summary_op_loss,
             'op_assign_testing_psnr': op_assign_testing_psnr,
             'summary_op_testing_psnr': summary_op_testing_psnr,
             'ph_testing_psnr': ph_testing_psnr
             }
    return model


def build_recon_network_dual(list_n_features_encoder=[],
                             list_layer_type_encoder=[],
                             list_n_features_decoder=[],
                             list_layer_type_decoder=[],
                             weight_dict=[],
                             img_h=-1,
                             img_w=-1,
                             img_chs=31,
                             SINGLE_CASSI=False,
                             list_shift=[],
                             n_features_in_code=64,
                             rho=1e-01,
                             lambda_alpha_fidelity=1e-01,
                             learning_rate=1e-01):

    n_convs_encoder = len(list_layer_type_encoder)
    n_convs_decoder = len(list_layer_type_decoder)
    is_reusable = True
    #########################################################
    # Set placeholders
    #########################################################
    # the shape should be (batchsize, psize, psize, n_channels
    # sparse_code_alpha = tf.placeholder(params.TF_DATA_TYPE, name='alpha')
    code_shape = [1, img_h, img_w, n_features_in_code]
    img_gradient_shape = [1, img_h - 1, img_w - 1, img_chs, 2]
    # code_gradient_shape = [1, img_h - 1, img_w - 1, n_features_in_code]

    img_gt = tf.placeholder(params.TF_DATA_TYPE, name='img_gt')
    mask3d = tf.placeholder(params.TF_DATA_TYPE, name='mask3d')

    # xk = tf.Variable(tf.abs(tf.random_normal(shape=code_shape))*0.001, name='xk')
    xk = tf.Variable(tf.ones(shape=code_shape) * 0.0001, name='xk')
    xk_ph = tf.placeholder(params.TF_DATA_TYPE, name='xk_ph')
    op_assign_xk = xk.assign(xk_ph)

    # admm_zk = tf.Variable(tf.abs(tf.random_normal(shape=code_shape))*0.512, name='admm_zk')
    zk = tf.Variable(tf.ones(shape=img_gradient_shape) * 0.0001, name='zk', trainable=False)
    zk_ph = tf.placeholder(params.TF_DATA_TYPE, name='zk_ph')
    op_assign_zk = zk.assign(zk_ph)

    uk = tf.Variable(tf.ones(shape=img_gradient_shape) * 0.0001, name='uk', trainable=False)
    uk_ph = tf.placeholder(params.TF_DATA_TYPE, name='uk_ph')
    op_assign_uk = uk.assign(uk_ph)
    #########################################################
    # Build the decoder
    #########################################################
    layer_name_base = 'decoder'
    response = xk

    for l in xrange(n_convs_decoder):
        layer_name = layer_name_base + '-conv' + str(l)

        list_stride = [1, 1, 1, 1]
        pad = 'SAME'
        with tf.variable_scope(layer_name, reuse=is_reusable):
            key_weight = layer_name + "/weight:0"
            weight_val = weight_dict[key_weight]
            conv_weight = tf.constant(weight_val, name="weight")
            key_bias = layer_name + "/bias:0"
            bias_val = weight_dict[key_bias]
            conv_bias = tf.constant(bias_val, name="bias")
            conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
            response = tf.nn.bias_add(conv, conv_bias)
            response = tf.nn.relu(response)

    # img_recon = response
    img_recon = tf.identity(response, name='img_recon')
    # apply coded mask
    img_masked = tf.mul(img_recon, mask3d, name='masking')
    if SINGLE_CASSI:
        list_chs = []
        for ch in xrange(img_chs):
            shift_val = list_shift[ch]
            img_masked_ch = img_masked[:, :, :, ch]
            tensor_left = img_masked_ch[:, :, (img_w - shift_val):]
            tensor_right = img_masked_ch[:, :, 0:(img_w - shift_val)]
            tensor_concat = tf.concat(2, [tensor_left, tensor_right])
            list_chs.append(tensor_concat)
        img_masked = tf.pack(list_chs, axis=3)


    # projection to 2D
    img_prj = tf.reduce_sum(img_masked, 3)
    # normlaize
    img_prj = img_prj / np.float(img_chs)


    #########################################################
    # Build the encoder
    #########################################################
    layer_name_base = 'encoder'
    response = img_recon
    for l in xrange(n_convs_encoder):
        layer_name = layer_name_base + '-conv' + str(l)
        list_stride = [1, 1, 1, 1]
        pad = 'SAME'

        with tf.variable_scope(layer_name, reuse=is_reusable):
            key_weight = layer_name + "/weight:0"
            weight_val = weight_dict[key_weight]
            conv_weight = tf.constant(weight_val, name="weight")
            key_bias = layer_name + "/bias:0"
            bias_val = weight_dict[key_bias]
            conv_bias = tf.constant(bias_val, name="bias")
            conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
            response = tf.nn.bias_add(conv, conv_bias)
            if l == (n_convs_encoder - 1):
                response = tf.identity(response)
            else:
                response = tf.nn.relu(response)

    xk_from_encoder = tf.identity(response, name='sparse_code_alpha')

    #########################################################
    # Build loss functions
    #########################################################
    # loss data
    diff = img_prj - img_gt
    loss_data = 0.5 * tf.reduce_mean(tf.square(diff))

    # alpha fidelity
    diff_xk = xk - xk_from_encoder
    loss_alpha_fidelity = 1.0 * lambda_alpha_fidelity * 0.5 * tf.reduce_mean(tf.square(diff_xk))

    # loss ADMM
    G_xk_vertical = img_recon[:, 1:, :, :] - img_recon[:, :-1, :, :]
    G_xk_horizontal = img_recon[:, :, 1:, :] - img_recon[:, :, :-1, :]
    G_xk_vertical = G_xk_vertical[:, :, :-1, :]
    G_xk_horizontal = G_xk_horizontal[:, :-1, ::]
    G_xk = tf.pack([G_xk_vertical, G_xk_horizontal], axis=4)
    # G_xk = tf.square(G_xk_vertical) + tf.square(G_xk_horizontal)
    diff = G_xk - zk + uk
    # diff = xk - zk + uk
    loss_admm = 1.0 * 0.5 * rho * tf.reduce_mean(tf.square(diff))


   # regularization on code
    # tv on featrue
    #######################################################################
    # IGNORE
    #######################################################################
    n_total_pixels = img_h * img_w * img_chs
    loss_tv_feature = 0.0 * 1e-02 * (tf.nn.l2_loss(xk[:, 1:, :, :] - xk[:, :img_h - 1, :, :]) / n_total_pixels
                                     + tf.nn.l2_loss(xk[:, :, 1:, :] - xk[:, :, :img_w - 1, :]) / n_total_pixels)

    # tv on recon image
    # -02
    loss_tv_recon = 0.0 * 1e-03 * (
        tf.nn.l2_loss(img_recon[:, 1:, :, :] - img_recon[:, :img_h - 1, :, :]) / n_total_pixels
        + tf.nn.l2_loss(img_recon[:, :, 1:, :] - img_recon[:, :, :img_w - 1, :]) / n_total_pixels)

    # tv for spectral dimension
    loss_tv_spectrum \
        = 0.0 * 1e-08 * 1.0 * tf.nn.l2_loss(img_recon[:, :, :, 1:] - img_recon[:, :, :, :img_chs - 1]) / n_total_pixels

    # reg
    loss_reg_mag = 0.0 * 1e-03 * 0.5 * tf.reduce_mean((tf.square(xk)))
    #######################################################################


    loss = loss_data + loss_admm + loss_alpha_fidelity
           #+ loss_tv_feature + loss_tv_recon + loss_tv_spectrum \
           #+ loss_reg_mag
    loss = loss * 1e+00

    # data term
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # add summary
    loss_summary = tf.summary.scalar('loss', loss)
    summary_op_loss = tf.summary.merge([loss_summary])

    testing_psnr = tf.Variable(0.0, name='var_testing_psnr')
    ph_testing_psnr = tf.placeholder(dtype=tf.float32)
    op_assign_testing_psnr = tf.assign(testing_psnr, ph_testing_psnr)
    testing_psnr_summary = tf.summary.scalar('testing psnr', testing_psnr)
    summary_op_testing_psnr = tf.summary.merge([testing_psnr_summary])


    #########################################################
    # Return model
    #########################################################
    model = {'xk': xk,
             'op_assign_xk': op_assign_xk,
             'xk_ph': xk_ph,
             'xk_from_encoder': xk_from_encoder,
             'zk': zk,
             'op_assign_zk': op_assign_zk,
             'zk_ph': zk_ph,
             'uk': uk,
             'op_assign_uk': op_assign_uk,
             'uk_ph': uk_ph,
             'img_gt': img_gt,
             'mask3d': mask3d,
             'img_recon': img_recon,
             'img_prj': img_prj,
             'loss': loss,
             'loss_data': loss_data,
             'loss_alpha_fidelity': loss_alpha_fidelity,
             'loss_admm': loss_admm,
             'loss_tv_feature': loss_tv_feature,
             'loss_tv_recon': loss_tv_recon,
             'loss_tv_spec': loss_tv_spectrum,
             'loss_reg_mag': loss_reg_mag,
             'optimizer': optimizer,
             'summary_op_loss': summary_op_loss,
             'op_assign_testing_psnr': op_assign_testing_psnr,
             'summary_op_testing_psnr': summary_op_testing_psnr,
             'ph_testing_psnr': ph_testing_psnr
             }
    return model
