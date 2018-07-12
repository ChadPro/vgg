# -- coding: utf-8 --

import numpy as np
import tensorflow as tf

# input & output param
IMAGE_SIZE = 224
NUM_CHANNELS = 3
STDDEV = 0.01
IMAGENET_MEAN = [122.173, 116.150, 103.504] #bgr
DEFAULT_OUTPUT_NODE = 1000
BN_DECAY = 0.9
ACTIVATION = tf.nn.relu

def vgg_conv(net_input, filter_size, strides, padding, scope, trainable=True):
    with tf.variable_scope(scope):
        conv_weights = tf.get_variable("weights", filter_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV), trainable=trainable)
        conv_biases = tf.get_variable("biases", filter_size[-1], initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d(net_input, conv_weights, strides=strides, padding=padding)
        net = tf.nn.relu(tf.nn.bias_add(net, conv_biases))
    return net, conv_weights, conv_biases

def vgg_max_pool(net_input, ksize, strides, padding, scope):
    with tf.name_scope(scope):
        net = tf.nn.max_pool(net_input, ksize=ksize, strides=strides, padding=padding)
    return net

def vgg_fc(net_input,fc_size,train,regularizer,scope):
    with tf.variable_scope(scope):
        fc_weights = tf.get_variable("weights", fc_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularizer != None:
            tf.add_to_collection('regular_losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("biases", fc_size[-1], initializer=tf.constant_initializer(STDDEV))
        fc = tf.nn.relu(tf.matmul(net_input, fc_weights) + fc_biases)
        if train == "train":
            fc = tf.nn.dropout(fc, 0.5)
    return fc

def vgg_logit(net_input, fc_size, regularizer, scope):
    with tf.variable_scope(scope):
        fc_weights = tf.get_variable("weights", fc_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularizer != None:
            tf.add_to_collection('regular_losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("biases", fc_size[-1], initializer=tf.constant_initializer(STDDEV))
        fc = tf.matmul(net_input, fc_weights) + fc_biases
    return fc

def vgg_net(inputs, \
            num_classes=DEFAULT_OUTPUT_NODE, \
            is_training=True, \
            train="train", \
            regularizer=None, \
            reuse=None, \
            fine_tune=False, \
            scope='vgg11_224'):
    # rgb --> bgr
    rgb_scaled = inputs
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    bgr = tf.concat(axis=3, values=[blue - IMAGENET_MEAN[0], green - IMAGENET_MEAN[1], red - IMAGENET_MEAN[2], ])
    assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

    weights_trainable = True
    if fine_tune:
        weights_trainable = False

    nets_dict = {}
    variables_restore = []
    with tf.variable_scope(scope):
        with tf.variable_scope('part_1'):
            net, w1_1, b1_1 = vgg_conv(bgr, filter_size=[3,3,3,64], strides=[1,1,1,1], padding='SAME', scope='conv_1', trainable=weights_trainable)
            variables_restore.append(w1_1)
            variables_restore.append(b1_1)
            net, w1_2, b1_2 = vgg_conv(bgr, filter_size=[3,3,64,64], strides=[1,1,1,1], padding='SAME', scope='conv_2', trainable=weights_trainable)
            variables_restore.append(w1_2)
            variables_restore.append(b1_2)

        net = vgg_max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', scope='max_pool_1')

        with tf.variable_scope('part_2'):
            net, w2_1, b2_1 = vgg_conv(net, filter_size=[3,3,64,128], strides=[1,1,1,1], padding='SAME', scope='conv_1', trainable=weights_trainable)
            variables_restore.append(w2_1)
            variables_restore.append(b2_1)
            net, w2_2, b2_2 = vgg_conv(net, filter_size=[3,3,128,128], strides=[1,1,1,1], padding='SAME', scope='conv_2', trainable=weights_trainable)
            variables_restore.append(w2_2)
            variables_restore.append(b2_2)

        net = vgg_max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', scope='max_pool_2')

        with tf.variable_scope('part_3'):
            net, w3_1, b3_1 = vgg_conv(net, filter_size=[3,3,128,256], strides=[1,1,1,1], padding='SAME', scope='conv_1', trainable=weights_trainable)
            net, w3_2, b3_2 = vgg_conv(net, filter_size=[3,3,256,256], strides=[1,1,1,1], padding='SAME', scope='conv_2', trainable=weights_trainable)
            net, w3_3, b3_3 = vgg_conv(net, filter_size=[3,3,256,256], strides=[1,1,1,1], padding='SAME', scope='conv_3', trainable=weights_trainable)
            net, w3_4, b3_4 = vgg_conv(net, filter_size=[3,3,256,256], strides=[1,1,1,1], padding='SAME', scope='conv_4', trainable=weights_trainable)
            variables_restore.append(w3_1)
            variables_restore.append(b3_1)
            variables_restore.append(w3_2)
            variables_restore.append(b3_2)
            variables_restore.append(w3_3)
            variables_restore.append(b3_3)
            variables_restore.append(w3_4)
            variables_restore.append(b3_4)

        net = vgg_max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', scope='max_pool_3')

        with tf.variable_scope('part_4'):
            net, w4_1, b4_1 = vgg_conv(net, filter_size=[3,3,256,512], strides=[1,1,1,1], padding='SAME', scope='conv_1', trainable=weights_trainable)
            net, w4_2, b4_2 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_2', trainable=weights_trainable)
            net, w4_3, b4_3 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_3', trainable=weights_trainable)
            net, w4_4, b4_4 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_4', trainable=weights_trainable)
            variables_restore.append(w4_1)
            variables_restore.append(b4_1)
            variables_restore.append(w4_2)
            variables_restore.append(b4_2)
            variables_restore.append(w4_3)
            variables_restore.append(b4_3)
            variables_restore.append(w4_4)
            variables_restore.append(b4_4)

        net = vgg_max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', scope='max_pool_4')

        with tf.variable_scope('part_5'):
            net, w5_1, b5_1 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_1', trainable=weights_trainable)
            net, w5_2, b5_2 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_2', trainable=weights_trainable)
            net, w5_3, b5_3 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_3', trainable=weights_trainable)
            net, w5_4, b5_4 = vgg_conv(net, filter_size=[3,3,512,512], strides=[1,1,1,1], padding='SAME', scope='conv_4', trainable=weights_trainable)
            variables_restore.append(w5_1)
            variables_restore.append(b5_1)
            variables_restore.append(w5_2)
            variables_restore.append(b5_2)
            variables_restore.append(w5_3)
            variables_restore.append(b5_3)
            variables_restore.append(w5_4)
            variables_restore.append(b5_4)

        net = vgg_max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', scope='max_pool_5')

        pool_shape = net.get_shape().as_list()
        nodes_num = pool_shape[1]*pool_shape[2]*pool_shape[3]
        net_reshaped = tf.reshape(net, [pool_shape[0], nodes_num])

        with tf.variable_scope('part_6'):
            net = vgg_fc(net_reshaped, fc_size=[nodes_num,4096], train=train, regularizer=regularizer, scope="fc_1")
            net = vgg_fc(net, fc_size=[4096,4096], train=train, regularizer=regularizer, scope="fc_2")
            logit = vgg_logit(net, fc_size=[4096,num_classes], regularizer=regularizer, scope="fc_3")

    return logit, variables_restore