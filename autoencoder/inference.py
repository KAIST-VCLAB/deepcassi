




import tensorflow as tf
import autoencoder.model as ae_model

def infer_ae(data,
             list_n_features_encoder=ae_model.default_n_features_encoder,
             list_layer_type_encoder=ae_model.default_layer_type_encoder,
             list_n_features_decoder=ae_model.default_n_features_decoder,
             list_layer_type_decoder=ae_model.default_layer_type_decoder,
             filename_model=''):

    #########################################################
    # Generate the model
    #########################################################
    model = ae_model.build_convoultiona_ae(list_n_features_encoder=list_n_features_encoder,
                                  list_layer_type_encoder=list_layer_type_encoder,
                                  list_n_features_decoder=list_n_features_decoder,
                                  list_layer_type_decoder=list_layer_type_decoder)

    x_data_node = model['x_data_node']
    x_data_predicted_node = model['x_data_predicted_node']
    ae_codes = model['code']
    saver = model['saver']

    #########################################################
    # Perform inference
    #########################################################
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        saver.restore(sess, filename_model)
        print "Model restored from file: %s" % filename_model

        feed_dict = {x_data_node: data}
        recon, code\
            = sess.run([x_data_predicted_node, ae_codes], feed_dict=feed_dict)

    tf.reset_default_graph()
    return recon, code