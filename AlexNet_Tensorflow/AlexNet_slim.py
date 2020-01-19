#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : AlexNet_slim.py
# @ Description: reference http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#                          https://github.com/kratzert/finetune_alexnet_with_tensorflow
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/18 PM 3:07
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import xavier_initializer, xavier_initializer_conv2d


class AlexNet():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, batch_size, learning_rate=0.001, keep_prob=0.5, weight_decay=0.00005,
                 is_pretrain=False,):
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay

        self._R_MEAN = 123.68
        self._G_MEAN = 116.779
        self._B_MEAN = 103.939

        self.is_pretrain = is_pretrain
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
        self.raw_input_data = self.mean_subtraction(self.raw_input_data, means=[self._R_MEAN,
                                                                                self._G_MEAN,
                                                                                self._B_MEAN])
        self.raw_input_data = self.convert_rgb_to_bgr(self.raw_input_data)
        # y [None,num_classes]
        self.raw_input_label = tf.placeholder(tf.float32, shape=[None, self.num_classes], name="class_label")
        self.is_training = tf.compat.v1.placeholder_with_default(input=False, shape=(), name='is_training')

        self.global_step = tf.train.create_global_step()
        # self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        # self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # logits
        self.logits =  self.inference(inputs=self.raw_input_data, name='alexnet')
        # computer loss value
        self.loss = self.losses(labels=self.raw_input_label, logits=self.logits, name='loss')
        # train operation
        self.train = self.training(self.learning_rate, self.global_step)
        self.accuracy = self.evaluate_batch(logits=self.logits, labels=self.raw_input_label) / self.batch_size

    def inference(self, inputs, name):
        """
        vgg16 inference
        construct static map
        :param input_op:
        :return:
        """
        # inputs = tf.image.per_image_standardization(inputs)
        self.parameters = []
        # inputs /= 255.
        with tf.variable_scope(name, reuse=None) as sc:
            prop = self.alexnet(inputs=inputs,
                               num_classes= self.num_classes,
                               is_training = self.is_training,
                               weight_decay = self.weight_decay,
                               keep_prob = self.keep_prob,
                               scope=sc)
            return prop

    def alexnet(self, inputs,
                num_classes=None,
                is_training=True,
                weight_decay=0.0005,
                keep_prob=0.5,
                scope='alexnet'):
        with tf.variable_scope(scope, 'alexnet', [inputs]) as sc:
            with slim.arg_scope(self.alexnet_arg_scope(weight_decay=weight_decay)):
                # 227 * 227 * 3
                net = self.conv_layer(inputs, 96, [11, 11], 4, padding='VALID', name='conv1')
                # 55 * 55 * 64
                # Local Response Normalization
                lrn1 = tf.nn.lrn(net, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
                net = slim.max_pool2d(lrn1, [3, 3], 2, padding='VALID', scope='pool1')

                net = self.conv_layer(net, 256, [5, 5], groups=2, name='conv2')
                lrn2 = tf.nn.lrn(net, depth_radius=2, alpha=2e-05, beta=0.75, bias=1.0)
                net = slim.max_pool2d(lrn2, [3, 3], 2, scope='pool2')

                net = self.conv_layer(net, 384, [3, 3], name='conv3')
                net = self.conv_layer(net, 384, [3, 3], groups=2, name='conv4')

                net = self.conv_layer(net, 256, [3, 3], groups=2, name='conv5')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

                # flatten and full connect layer
                net = tf.reshape(net, shape=[-1, 6 * 6 * 256], name='flattened')
                net = slim.fully_connected(net, num_outputs=4096, scope='fc6')
                net = slim.dropout(net, keep_prob=keep_prob)
                net = slim.fully_connected(net, num_outputs=4096, scope='fc7')
                net = slim.dropout(net, keep_prob=keep_prob)

                net = slim.fully_connected(net, num_outputs=num_classes, activation_fn=None, scope='fc8')
                logits = slim.softmax(net, scope='logits')

                return logits

    def training(self, learnRate, globalStep):
        """
        train operation
        :param learnRate:
        :param globalStep:
        :param args:
        :return:
        """
        # define trainable variable
        trainable_variable = None
        # trainable_scope = self.trainable_scope
        # trainable_scope = ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8_1', 'vgg_16/fc8_2']
        trainable_scope = []
        if self.is_pretrain and trainable_scope:
            trainable_variable = []
            for scope in trainable_scope:
                variables = tf.model_variables(scope=scope)
                [trainable_variable.append(var) for var in variables]

        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        train_op = tf.train.AdamOptimizer(learnRate).minimize(self.loss, global_step=globalStep,
                                                              var_list=trainable_variable)
        return train_op

    def losses(self, logits, labels, name):
        """
        loss function
        :param logits:
        :param labels:
        :return:
        """
        with tf.name_scope(name):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='entropy')
            return tf.reduce_mean(input_tensor=cross_entropy, name='loss')

    def evaluate_batch(self, logits, labels):
        """
        evaluate one batch correct num
        :param logits:
        :param label:
        :return:
        """
        correct_predict = tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1))
        return tf.reduce_sum(tf.cast(correct_predict, dtype=tf.int32))

    def fill_feed_dict(self, image_feed, label_feed, is_training):
        feed_dict = {
            self.raw_input_data: image_feed,
            self.raw_input_label: label_feed,
            self.is_training: is_training
        }
        return feed_dict

    def conv_layer(self, inputs, num_outputs, kernel_size, stride=1, name=None, padding='SAME', use_bias=True, groups=1):
        """Create a convolution layer.
        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(inputs.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda inputs, kernel: tf.nn.conv2d(inputs, kernel,
                                             strides=[1, stride, stride, 1],
                                             padding=padding)

        with tf.variable_scope(name,) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights',
                                      shape=[kernel_size[0], kernel_size[1], input_channels / groups, num_outputs],
                                      initializer=xavier_initializer_conv2d(),
                                      trainable=True)
            # tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, weights)

            biases = tf.get_variable('biases', shape=[num_outputs])
            # tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, biases)

        if groups == 1:
            # net = slim.conv2d(inputs, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride, padding=padding,
            #                   scope=name)
            net = convolve(inputs, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(value=inputs, axis=3, num_or_size_splits=groups)
            weight_groups = tf.split(value=weights, axis=3, num_or_size_splits=groups)

            output_groups = [convolve(input, kernel) for input, kernel in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            net = tf.concat(axis=3, values=output_groups)
        if use_bias:
            # Add biases
            bias = tf.reshape(tf.nn.bias_add(net, biases), tf.shape(net))
            # Apply relu function
            net = tf.nn.relu(bias, name=scope.name)

        return net

    def load_pretrain_model(self, sess, model_path):
        """

        :param sess:
        :param model_path:
        :return:
        """
        weights_dict = np.load(model_path, encoding='bytes', allow_pickle=True).item()
        weights_dict = dict(weights_dict)

        custom_scope = ['fc8']

        for op_name in weights_dict:
            if op_name not in custom_scope:
                for data in weights_dict[op_name]:
                    with tf.variable_scope('alexnet/' + op_name, reuse=True):
                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=True)
                            sess.run(var.assign(data))
                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=True)
                            sess.run(var.assign(data))
        print('+++++++++++++++++++Successful load all variable+++++++++++++++++++')

    def alexnet_arg_scope(self, weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def mean_subtraction(self, image, means):
        """
        subtract the means form each image channel (white image)
        :param image:
        :param mean:
        :return:
        """
        num_channels = image.get_shape()[-1]
        image = tf.cast(image, dtype=tf.float32)
        channels = tf.split(value=image, num_or_size_splits=num_channels, axis=3)
        for n in range(num_channels):
            channels[n] -= means[n]
        return tf.concat(values=channels, axis=3, name='concat_channel')

    def convert_rgb_to_bgr(self, image):
        """
        image process
        :param image:
        :return:
        """
        return image[:, :, :, ::-1]



