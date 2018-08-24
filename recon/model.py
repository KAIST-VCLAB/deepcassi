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

def build_encoder_ae(list_n_features_encoder=[],
                     list_layer_type_encoder=[],
                     is_trainable=False,
                     weight_dict={}):
    n_convs_encoder = len(list_layer_type_encoder)
    is_reusable = True

    #########################################################
    # Set placeholders
    #########################################################
    # the shape should be (batchsize, psize, psize, n_channels
    x_data_node = tf.placeholder(params.TF_DATA_TYPE, name='data')

    #########################################################
    # Build the encoder
    #########################################################
    layer_name_base = 'encoder'
    response = x_data_node
    for l in range(n_convs_encoder):
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

    sparse_code_alpha = tf.identity(response, name='sparse_code_alpha')
    #########################################################
    # Return model
    #########################################################
    model = {'x_data_node': x_data_node,
             'sparse_code_alpha': sparse_code_alpha}
    return model


def build_decoder_ae(list_n_features_decoder=[],
                     list_layer_type_decoder=[],
                     is_trainable=False,
                     weight_dict={}):
    n_convs_decoder = len(list_layer_type_decoder)
    is_resuable = True
    #########################################################
    # Set placeholders
    #########################################################
    # the shape should be (batchsize, psize, psize, n_channels
    sparse_code_alpha = tf.placeholder(params.TF_DATA_TYPE, name='alpha')

    #########################################################
    # Build the decoder
    #########################################################
    layer_name_base = 'decoder'
    response = sparse_code_alpha

    for l in range(n_convs_decoder):
        layer_name = layer_name_base + '-conv' + str(l)

        list_stride = [1, 1, 1, 1]
        pad = 'SAME'
        with tf.variable_scope(layer_name, reuse=is_resuable):
            key_weight = layer_name + "/weight:0"
            weight_val = weight_dict[key_weight]
            conv_weight = tf.constant(weight_val, name="weight")
            key_bias = layer_name + "/bias:0"
            bias_val = weight_dict[key_bias]
            conv_bias = tf.constant(bias_val, name="bias")
            conv = tf.nn.conv2d(response, conv_weight, strides=list_stride, padding=pad)
            response = tf.nn.bias_add(conv, conv_bias)
            response = tf.nn.relu(response)

    img_recon = tf.identity(response, name='img_recon')

    #########################################################
    # Return model
    #########################################################
    model = {'sparse_code_alpha': sparse_code_alpha,
             'img_recon': img_recon,
             }
    return model
