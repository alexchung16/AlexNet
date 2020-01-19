#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : AlexNet_load_model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/19 上午10:51
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from AlexNet_Tensorflow.AlexNet_slim import AlexNet

alexnet_model = '/home/alex/Documents/pretraing_model/alexnet/alexnet.npy'

if __name__ == "__main__":
    weights_dict = np.load(alexnet_model, encoding='bytes', allow_pickle=True).item()
    weights_dict = dict(weights_dict)

    for layer, weights in weights_dict.items():
        print(layer)
        print('weight: {0}'.format(weights[0].shape))
        print('bias: {0}'.format(weights[1].shape))
        print('-' * 40)

    alex_net = AlexNet(input_shape=(227, 227, 3),
                       num_classes=2,
                       batch_size=10,
                       learning_rate=0.01)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer(),
                       tf.initialize_all_variables())
    with tf.Session() as sess:
        sess.run(init_op)

        # custom scope weight will not be load
        custom_scope = ['fc8']

        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.op.name, var.shape)
        for op_name in weights_dict:
            if op_name not in custom_scope:
                for data in weights_dict[op_name]:
                    with tf.variable_scope('alexnet/'+op_name, reuse=True):
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=True)
                            sess.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=True)
                            sess.run(var.assign(data))

        with tf.variable_scope('alexnet/conv1', reuse=True):
            model_variable = tf.get_variable('biases')
            print(model_variable)













