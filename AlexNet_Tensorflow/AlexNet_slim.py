#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : AlexNet_slim.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/18 下午3:07
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim


class AlexNet():
    """
    VGG16 model
    """
    def __init__(self, input_shape, num_classes, batch_size, decay_rate, learning_rate,keep_prob=0.5,
                 weight_decay=0.00005, num_samples_per_epoch=None, num_epoch_per_decay=None, is_pretrain=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.decay_steps = int(num_samples_per_epoch / batch_size * num_epoch_per_decay)
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        # self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.weight_decay = weight_decay

        self.is_pretrain = is_pretrain
        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94
        # self.initializer = tf.random_normal_initializer(stddev=0.1)
        # add placeholder (X,label)
        self.raw_input_data = tf.placeholder(tf.float32, shape=[None, input_shape[0], input_shape[1], input_shape[2]],
                                             name="input_images")
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
                net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
                net = slim.conv2d(net, 192, [5, 5], scope='conv2')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = slim.conv2d(net, 384, [3, 3], scope='conv3')
                net = slim.conv2d(net, 384, [3, 3], scope='conv4')
                net = slim.conv2d(net, 256, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

                with slim.arg_scope([slim.conv2d],
                                    weights_initializer= tf.truncated_normal_initializer(0.0, 0.005),
                                    biases_initializer=tf.constant_initializer(0.1)):
                    net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                                      scope='fc6')
                    net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout6')

                    net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                    net = slim.dropout(net, keep_prob, is_training=is_training, scope='dropout7')

                    net = slim.conv2d(net, num_classes, [1, 1],  activation_fn=None,  normalizer_fn=None,
                                      biases_initializer=tf.zeros_initializer(), scope='fc8')
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    logits = slim.softmax(logits=net, scope='softmax')
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

        learning_rate = tf.train.exponential_decay(learning_rate=learnRate, global_step=globalStep,
                                                   decay_steps=self.decay_steps, decay_rate=self.decay_rate,
                                                   staircase=False)
        # according to use request of slim.batch_norm
        # update moving_mean and moving_variance when training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=globalStep,
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

    def alexnet_arg_scope(self, weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc


