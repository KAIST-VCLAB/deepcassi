


import tensorflow as tf
import numpy as np
import cv2
import scipy.io as sio
from datetime import datetime

import modulation
import visualizer.drawer as vis
import recon.model as model_recon_basics
import recon.snapshot.model as model_recon_cassi
import recon.misc as misc
import autoencoder.model as ae_model

def recon_snapshot(img_snapshot=[],
                   img_mask=[],
                   gt_hs=[],
                   out_filename=[],
                   img_n_chs=31,
                   list_shift =  modulation.shift_ours,
                   list_shift_y = np.zeros(shape=(1,31)),
                   SINGLE_CASSI=False,
                   SSCSI=False,
                   param_rho=1e-01,
                   param_sparsity = 0.01,
                   param_lambda_alpha_fidelity=1e-01,
                   param_learning_rate=1e-01,
                   n_iters_ADMM=15,
                   n_iters_ADAM = 100,
                   ENABLE_ALPHA_FIDELITY=True,
                   list_n_features_encoder=ae_model.default_n_features_encoder,
                   list_layer_type_encoder=ae_model.default_layer_type_encoder,
                   list_n_features_decoder=ae_model.default_n_features_decoder,
                   list_layer_type_decoder=ae_model.default_layer_type_decoder,
                   filename_model='',
                   summary_dir='./summary_recon',
                   do_summarize=False,
                   GPU_ID=''):

    #########################################################
    # check if correct inputs are given
    #########################################################
    if img_snapshot == [] or img_mask == []:
        print 'Not enough input images are given! Terminating....'
        exit()

    #########################################################
    # initialize
    #########################################################
    param_lambda = param_sparsity * param_rho
    img_h, img_w = img_snapshot.shape
    n_features_in_code = list_n_features_encoder[-1]
    img_coded = np.expand_dims(img_snapshot, axis=0)

    if SSCSI:
        mask3d = modulation.shift_random_mask(img_mask, shift=0.1)
    else:
        mask3d = modulation.generate_shifted_mask_cube(img_mask, chs=img_n_chs,
                                                       shift_list=list_shift,
                                                       shift_list_y=list_shift_y,
                                                        SINGLE_CASSI=SINGLE_CASSI)
    mask3d_tf = np.zeros(shape=(1, img_h, img_w, img_n_chs),
                         dtype=np.float32)
    mask3d_tf[0, :, :, :] = mask3d

    if gt_hs != []:
        data = np.expand_dims(gt_hs, axis=0)

    #########################################################
    # build the autoencoder and load pre-trained weights
    #########################################################
    graph_ae = tf.Graph()
    with graph_ae.as_default():
        model_ae = ae_model.build_convoultiona_ae(list_n_features_encoder=list_n_features_encoder,
                                               list_layer_type_encoder=list_layer_type_encoder,
                                               list_n_features_decoder=list_n_features_decoder,
                                               list_layer_type_decoder=list_layer_type_decoder,
                                               is_trainable=False)
        tf_var_dict = {}
        weight_dict = {}
        for v in tf.global_variables():
            #print v.name
            tf_var_dict[v.name] = v

        # load trained weights
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, filename_model)
            for key in tf_var_dict.keys():
                tf_var = tf_var_dict[key]
                weight_value, = sess.run([tf_var])
                weight_dict[key] = weight_value



    #########################################################
    # build the decoder and encoder
    #########################################################
    graph_ae_separated = tf.Graph()
    with graph_ae_separated.as_default():
        model_encoder = model_recon_basics.build_encoder_ae(list_n_features_encoder=list_n_features_encoder,
                                                      list_layer_type_encoder=list_layer_type_encoder,
                                                      weight_dict=weight_dict)

        model_decoder = model_recon_basics.build_decoder_ae(list_n_features_decoder=list_n_features_decoder,
                                                      list_layer_type_decoder=list_layer_type_decoder,
                                                      weight_dict=weight_dict)

    #########################################################
    # build recon network
    #########################################################
    graph_recon_network = tf.Graph()
    with graph_recon_network.as_default():
        if ENABLE_ALPHA_FIDELITY:
            model_recon = model_recon_cassi.build_recon_network_dual(list_n_features_encoder=list_n_features_encoder,
                                                               list_layer_type_encoder=list_layer_type_encoder,
                                                               list_n_features_decoder=list_n_features_decoder,
                                                               list_layer_type_decoder=list_layer_type_decoder,
                                                               weight_dict=weight_dict,
                                                               img_h=img_h,
                                                               img_w=img_w,
                                                               img_chs=img_n_chs,
                                                               SINGLE_CASSI=SINGLE_CASSI,
                                                               list_shift=list_shift,
                                                               n_features_in_code=n_features_in_code,
                                                               rho=param_rho,
                                                               learning_rate=param_learning_rate,
                                                               lambda_alpha_fidelity=param_lambda_alpha_fidelity)
        else:
            model_recon = model_recon_cassi.build_recon_network(list_n_features_decoder=list_n_features_decoder,
                                                          list_layer_type_decoder=list_layer_type_decoder,
                                                          weight_dict=weight_dict,
                                                          img_h=img_h,
                                                          img_w=img_w,
                                                          img_chs=img_n_chs,
                                                          n_features_in_code=n_features_in_code,
                                                          rho=param_rho,
                                                          learning_rate=param_learning_rate)

    xk = model_recon['xk']
    op_assign_xk = model_recon['op_assign_xk']
    xk_ph = model_recon['xk_ph']
    zk = model_recon['zk']
    op_assign_zk = model_recon['op_assign_zk']
    zk_ph = model_recon['zk_ph']
    uk = model_recon['uk']
    op_assign_uk = model_recon['op_assign_uk']
    uk_ph = model_recon['uk_ph']
    img_gt = model_recon['img_gt']
    img_recon = model_recon['img_recon']
    img_prj = model_recon['img_prj']
    mask3d_ph = model_recon['mask3d']
    loss = model_recon['loss']
    loss_data = model_recon['loss_data']
    loss_admm = model_recon['loss_admm']
    loss_tv_feature = model_recon['loss_tv_feature']
    loss_tv_recon = model_recon['loss_tv_recon']
    loss_tv_spec = model_recon['loss_tv_spec']
    loss_reg_mag = model_recon['loss_reg_mag']
    optimizer = model_recon['optimizer']
    summary_op_loss = model_recon['summary_op_loss']

    ph_testing_psnr = model_recon['ph_testing_psnr']
    op_assign_testing_psnr = model_recon['op_assign_testing_psnr']
    summary_op_testing_psnr = model_recon['summary_op_testing_psnr']

    if ENABLE_ALPHA_FIDELITY:
        xk_from_encoder = model_recon['xk_from_encoder']
        loss_alpha_fidelity = model_recon['loss_alpha_fidelity']

    #########################################################
    # do recon
    #########################################################

    with graph_recon_network.as_default():
        with tf.Session() as sess:
            # saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            summary_writer.flush()
            step = 0

            for i_admm in xrange(n_iters_ADMM):
                print 'ADMM - iteration: %d' % (i_admm)
                start_time = datetime.now()
                n_iters = n_iters_ADAM

                for i_nonlinear in xrange(n_iters):
                    if ENABLE_ALPHA_FIDELITY:
                        _, dual_encoder_code, \
                        l_total, l_data, l_admm, l_alpha_fidelity \
                            = sess.run([optimizer, xk_from_encoder,
                                        loss, loss_data, loss_admm, loss_alpha_fidelity],
                                       feed_dict={img_gt: img_coded,
                                                  mask3d_ph: mask3d_tf})

                        if i_nonlinear % 50 == 0:
                            print 'iter: %d, total_loss: %6e, data_loss: %6e, admm_loss: %6e,' \
                                  ' alpha_loss: %6e,' \
                                  % (i_nonlinear, l_total, l_data, l_admm, l_alpha_fidelity)

                        if i_nonlinear == n_iters - 1:
                            # update xk (code)
                            #sess.run(op_assign_xk, feed_dict={xk_ph: dual_encoder_code})
                            pass
                    else:
                        _, \
                        l_total, l_data, l_admm, \
                        l_tv_f, l_tv_recon, l_tv_spec, \
                        l_reg_mag \
                            = sess.run([optimizer,
                                        loss, loss_data, loss_admm,
                                        loss_tv_feature, loss_tv_recon, loss_tv_spec,
                                        loss_reg_mag],
                                       feed_dict={img_gt: img_coded,
                                                  mask3d_ph: mask3d_tf})

                        if i_nonlinear % 50 == 0:
                            print 'iter: %d, total_loss: %6e, data_loss: %6e, admm_loss: %6e, ' \
                                  ' tv_loss_f: %6e, tv_recon_loss: %6e, tv_spec_loss: %6e, ' \
                                  'reg_mag_loss: %6e' \
                                  % (i_nonlinear, l_total, l_data, l_admm,
                                     l_tv_f, l_tv_recon, l_tv_spec, l_reg_mag)



                    if i_nonlinear % 50 == 0:
                        if ENABLE_ALPHA_FIDELITY:
                            code, code_encoder, img_result = sess.run([xk, xk_from_encoder, img_recon])
                            # img_result_15 = vis.normalize_1ch(img_result[0, :, :, 15])
                            img_result_15 = img_result[0, :, :, 15]
                            temp_max = np.max(img_result)
                            temp_min = np.min(img_result)
                            img_result_15 = (img_result_15 - temp_min) / (temp_max - temp_min)

                            vis.imshow_with_zoom(GPU_ID + "recon data_tf", np.power(img_result_15, 1 / 2.2), scale=1.0)
                            #vis.visualize_sparse_code(code, rows=8, cols=8, title=GPU_ID + 'code_tf', scale=0.25)
                            #vis.visualize_sparse_code(code_encoder, rows=8, cols=8, title=GPU_ID + 'code_tf_encoder', scale=0.25)

                            img_result_resize = cv2.resize(img_result[0], dsize=(img_w / 5, img_h / 5))
                            img_result_resize = np.expand_dims(img_result_resize, 0)

                            if gt_hs != []:
                                data_resize = cv2.resize(data[0], dsize=(img_w / 5, img_h / 5))
                                data_resize = np.expand_dims(data_resize, 0)

                                vis.draw_the_comparison(img_result_resize, data_resize, title=GPU_ID + 'image recon comparison')

                                gt_15 = data[0, :, :, 15]
                                temp_max = np.max(gt_15)
                                temp_min = np.min(gt_15)
                                gt_15 = (gt_15 - temp_min) / (temp_max - temp_min)

                                vis.imshow_with_zoom(GPU_ID + "GT", np.power(gt_15, 1 / 2.2),
                                                     scale=1.0)
                                test_diff_sqr = np.square(img_result - data)
                                test_diff_sqr_avg = np.average(test_diff_sqr)
                                test_psnr = -10.0 * np.log10(test_diff_sqr_avg)
                                print 'test_psnr: %.4f' % (test_psnr)
                            else:
                                temp_max = np.max(img_result_resize)
                                temp_min = np.min(img_result_resize)
                                img_result_resize = (img_result_resize - temp_min) / (temp_max - temp_min)
                                vis.draw_the_comparison(img_result_resize,
                                                        title=GPU_ID + 'image recon comparison',
                                                        compute_psnr=False)
                        else:
                            code, img_prj_est, img_result \
                                = sess.run([xk, img_prj, img_recon],
                                           feed_dict={mask3d_ph: mask3d_tf})


                            img_result_15 = img_result[0, :, :, 15]
                            temp_max = np.max(img_result)
                            temp_min = np.min(img_result)
                            img_result_15 = (img_result_15 - temp_min) / (temp_max - temp_min)
                            vis.imshow_with_zoom("recon data_tf", np.power(img_result_15, 1 / 2.2), scale=1.0)
                            vis.visualize_sparse_code(code, rows=8, cols=8, title='code_tf', scale=0.25)
                            img_prj_est_vis = vis.normalize_1ch(img_prj_est[0])
                            vis.imshow_with_zoom('estimated SSCSI', img_prj_est_vis, scale=1.0)

                            img_result_resize = cv2.resize(img_result[0], dsize=(img_w / 5, img_h / 5))
                            img_result_resize = np.expand_dims(img_result_resize, 0)
                            if gt_hs != []:
                                data_resize = cv2.resize(data[0], dsize=(img_w / 5, img_h / 5))
                                data_resize = np.expand_dims(data_resize, 0)
                                vis.draw_the_comparison(img_result_resize, data_resize, title='image recon comparison')
                                test_diff_sqr = np.square(img_result - data)
                                test_diff_sqr_avg = np.average(test_diff_sqr)
                                test_psnr = -10.0 * np.log10(test_diff_sqr_avg)
                                print 'test_psnr: %.4f' % (test_psnr)
                            else:
                                temp_max = np.max(img_result_resize)
                                temp_min = np.min(img_result_resize)
                                img_result_resize = (img_result_resize - temp_min) / (temp_max - temp_min)
                                vis.draw_the_comparison(img_result_resize, img_result_resize,
                                                        title='image recon comparison',
                                                        compute_psnr=False)
                        cv2.waitKey(100)

                        # compute PSNR
                        if do_summarize == True and gt_hs != []:
                            code, img_result = sess.run([xk, img_recon])
                            test_diff_sqr = np.square(img_result - data)
                            test_diff_sqr_avg = np.average(test_diff_sqr)
                            psnr_val = -10.0 * np.log10(test_diff_sqr_avg)
                            sess.run(op_assign_testing_psnr, feed_dict={ph_testing_psnr: psnr_val})
                            summary_str = sess.run(summary_op_testing_psnr)
                            summary_writer.add_summary(summary_str, step)
                            summary_writer.flush()

                    step = step + 1

                # get variabels from TF
                xk1_val, img_val, zk_val, uk_val = sess.run([xk, img_recon, zk, uk])

                ############################################ for real image
                img_val[img_val > 1.0] = 1.0
                img_val[img_val < 0.0] = 0.0
                ############################################

                # update z
                np_G_xk1_val = misc.np_del_operator(img_val)
                zk1_val = misc.soft_threshold(np_G_xk1_val + uk_val, param_lambda, param_rho)

                #  update u
                uk1_val = uk_val + np_G_xk1_val - zk1_val

                # assign updated variables to TF
                sess.run([op_assign_zk], feed_dict={zk_ph: zk1_val})
                sess.run([op_assign_uk], feed_dict={uk_ph: uk1_val})

                end_time = datetime.now()
                elapsed_time = end_time - start_time
                print 'elapsed time for 1 ADMM iteration: ' + str(elapsed_time)

    cv2.waitKey(5000)

    wvls2b = np.arange(400, 701, 10)
    wvls2b = wvls2b.astype(np.float32)
    if out_filename != []:
        out_dict =  {'x_recon': np.squeeze(img_result),
                      'wvls2b': wvls2b}
        sio.savemat(out_filename, out_dict)
        # if gt_hs != []:
        #     out_dict_gt = {'x_recon': np.squeeze(gt_hs),
        #                    'wvls2b': wvls2b}
        #     sio.savemat('gt.mat', out_dict_gt)

    return img_result, wvls2b