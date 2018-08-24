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
import params


# 'c': convolutional
default_n_features_encoder = [params.N_VALID_SPECTRALS, 64, 64, 64, 64, 64, 64]
default_layer_type_encoder = ['c', 'c', 'c', 'c', 'c', 'c']
default_n_features_decoder = default_n_features_encoder[::-1]
default_layer_type_decoder = ['c', 'c', 'c', 'c', 'c', 'c']


def build_convoultiona_ae(list_n_features_encoder=default_n_features_encoder,
                          list_layer_type_encoder=default_layer_type_encoder,
                          list_n_features_decoder=default_n_features_decoder,
                          list_layer_type_decoder= default_layer_type_decoder,
                          is_trainable=True,
                          with_wd=True
                          ):
    #########################################################
    # Check dimensions
    #########################################################
    if list_n_features_encoder[-1] != list_n_features_decoder[0]:
        print('The output side of the encoder' \
              ' and the input size of decoder do not match')
        exit()
    if list_n_features_encoder[0] != list_n_features_decoder[-1]:
        print('The input side of the encoder' \
              ' and the output size of decoder do not match')
        exit()

    if (list_n_features_encoder[0] != params.N_VALID_SPECTRALS)\
            or(list_n_features_decoder[-1] != params.N_VALID_SPECTRALS):
        print('The input and the output sizes do not match with the n_channels')
        exit()

    if (len(list_n_features_encoder) != len(list_layer_type_encoder) + 1) \
            or (len(list_n_features_decoder) != len(list_layer_type_decoder) + 1):
        print('The input and the output sizes do not match with the n_channels')
        exit()

    n_convs_encoder = len(list_layer_type_encoder)
    n_convs_decoder = len(list_layer_type_decoder)

    #########################################################
    # Set placeholders
    #########################################################
    # the shape should be (batchsize, psize, psize, n_channels
    x_data_node = tf.placeholder(params.TF_DATA_TYPE, name='data')
    x_data_out_node = tf.placeholder(params.TF_DATA_TYPE, name='data')
    ksize = params.TF_CONV_KERNEL_SIZE

    # for weight decay
    conv_weight_list = []

    #########################################################
    # Build the encoder
    #########################################################
    layer_name_base = 'encoder'
    response = x_data_node
    for l in range(n_convs_encoder):
        l_type = list_layer_type_encoder[l]
        layer_name = layer_name_base + '-conv' + str(l)

        n_feature_prev = list_n_features_encoder[l]
        n_feature_next = list_n_features_encoder[l + 1]

        if l_type == 'c' or l_type == 'p':
            if l_type == 'c':
                list_stride = [1, 1, 1, 1]
                pad = 'SAME'
            else:
                list_stride = [1, 2, 2, 1]
                pad = 'SAME'

            with tf.variable_scope(layer_name):
                conv_weight = tf.get_variable("weight",
                                              shape=[ksize, ksize,
                                                     n_feature_prev,
                                                     n_feature_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable)
                conv_bias = tf.Variable(tf.zeros([n_feature_next],dtype=params.TF_DATA_TYPE),
                                        name='bias',
                                        trainable=is_trainable)
                conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
                response = tf.nn.bias_add(conv, conv_bias)

                if with_wd:
                    conv_weight_list.append(conv_weight)

                if l == (n_convs_encoder - 1):
                    response = tf.identity(response)
                else:
                    response = tf.nn.relu(response)
        else:
            print('A wrong layer type for the encoder')
            exit()

        if l == (n_convs_encoder - 1):
            response_code = response


    #########################################################
    # Build the decoder
    #########################################################
    layer_name_base = 'decoder'

    for l in range(n_convs_decoder):
        l_type = list_layer_type_decoder[l]
        layer_name = layer_name_base + '-conv' + str(l)

        n_feature_prev = list_n_features_decoder[l]
        n_feature_next = list_n_features_decoder[l + 1]

        if l_type == 'c':
            list_stride = [1, 1, 1, 1]
            pad = 'SAME'
            with tf.variable_scope(layer_name):
                conv_weight = tf.get_variable("weight",
                                              shape=[ksize, ksize,
                                                     n_feature_prev,
                                                     n_feature_next],
                                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              trainable=is_trainable)
                conv_bias = tf.Variable(tf.zeros([n_feature_next], dtype=params.TF_DATA_TYPE),
                                        name='bias',
                                        trainable=is_trainable)
                conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
                response = tf.nn.bias_add(conv, conv_bias)
                if l == (n_convs_decoder - 1):
                    response = tf.nn.relu(response)

                else:
                    response = tf.nn.relu(response)

                if with_wd:
                    conv_weight_list.append(conv_weight)
        else:
            print('A wrong layer type for the decoder')
            exit()

    x_data_predicted_node = response

    #########################################################
    # define the loss - data term
    #########################################################
    data_loss = tf.reduce_mean(tf.square(x_data_predicted_node - x_data_out_node), name='training_error')
    training_error = data_loss
    testing_error = tf.Variable(0.0, name='var_testing_err')
    ph_testing_error = tf.placeholder(dtype=tf.float32)
    op_assign_testing_error = tf.assign(testing_error, ph_testing_error)
    testing_psnr = tf.Variable(0.0, name='var_testing_psnr')
    ph_testing_psnr = tf.placeholder(dtype=tf.float32)
    op_assign_testing_psnr = tf.assign(testing_psnr, ph_testing_psnr)
    #########################################################
    # define the loss - weight decay term
    #########################################################
    if with_wd:
        weight_lambda = params.TF_WEIGHT_DECAY_LAMBDA
        weight_decay_term\
            = weight_lambda * tf.add_n([tf.nn.l2_loss(v) for v in conv_weight_list])
        weight_decay_term /= len(conv_weight_list)
        training_error += weight_decay_term

    #########################################################
    # Add summaries
    #########################################################
    training_error_summary = tf.summary.scalar('training error', training_error)
    data_loss_summary = tf.summary.scalar('data_loss', data_loss)
    testing_error_summary = tf.summary.scalar('testing error', testing_error)
    testing_psnr_summary = tf.summary.scalar('testing psnr', testing_psnr)

    if with_wd:
        weight_decay_error_summary = tf.summary.scalar('weight decay error', weight_decay_term)
        summary_op_weight_decay = tf.summary.merge([weight_decay_error_summary])


    #########################################################
    # Add saver
    #########################################################
    saver = tf.train.Saver()

    summary_op_training = tf.summary.merge([training_error_summary])
    summary_op_data_loss = tf.summary.merge([data_loss_summary])
    summary_op_testing = tf.summary.merge([testing_error_summary])
    summary_op_testing_psnr = tf.summary.merge([testing_psnr_summary])


    #########################################################
    # Return model
    #########################################################
    model = {'x_data_node': x_data_node,
             'x_data_out_node': x_data_out_node,
             'x_data_predicted_node': x_data_predicted_node,
             'code': response_code,
             'training_error': training_error,
             'data_loss': data_loss,
             'ph_testing_error': ph_testing_error,
             'op_assign_testing_error': op_assign_testing_error,
             'ph_testing_psnr': ph_testing_psnr,
             'op_assign_testing_psnr': op_assign_testing_psnr,
             'summary_op_training': summary_op_training,
             'summary_op_data_loss': summary_op_data_loss,
             'summary_op_testing': summary_op_testing,
             'summary_op_testing_psnr': summary_op_testing_psnr,
             'saver': saver,
             }

    if with_wd:
        model['weight_decay_term'] = weight_decay_term
        model['summary_op_weight_decay'] = summary_op_weight_decay

    return model
