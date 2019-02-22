# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import time
from nets import nets_factory
from tf_extend import vgg_acc

x = tf.ones([1,32,32,3])
vgg_net = nets_factory.get_network("vgg_cifar_net")
y = vgg_net.vgg_net(x, num_classes=10, is_training=True)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    r = sess.run(y)
    print(r.shape)