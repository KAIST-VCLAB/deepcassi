
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

    for l in xrange(n_convs_decoder):
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