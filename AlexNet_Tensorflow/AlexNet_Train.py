#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : AlexNet_Train.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/1/18 PM 4:38
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
# from VGG16.VGG16 import VGG16
from AlexNet_Tensorflow.AlexNet_slim import AlexNet
import numpy as np
from DataProcess.read_TFRecord import reader_tfrecord, dataset_tfrecord, get_num_samples
from tensorflow.python.framework import graph_util


# compatible GPU version problem
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# config
original_dataset_dir = '/home/alex/Documents/dataset/dogs_vs_cat_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_data_dir= os.path.join(tfrecord_dir, 'train')
val_data_dir= os.path.join(tfrecord_dir, 'test')

model_path = os.path.join(os.getcwd(), 'model')
model_name = os.path.join(model_path, 'alexnet.pb')
pretrain_model_dir = '/home/alex/Documents/pretraing_model/alexnet/alexnet.npy'
logs_dir = os.path.join(os.getcwd(), 'logs')


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 227, 'Number of height size.')
flags.DEFINE_integer('width', 227, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 2, 'Number of image class.')
flags.DEFINE_integer('batch_size', 128, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('keep_prop', 0.8, 'Number of probability that each element is kept.')
flags.DEFINE_float('weight_decay', 0.0005, 'Number of regular scale size')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'pretrain model dir.')
flags.DEFINE_string('train_data', train_data_dir, 'Directory to put the training data.')
flags.DEFINE_string('val_data', val_data_dir, 'Directory to put the validation data.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')


def predict(model_name, image_data, input_op_name, predict_op_name):
    """
    model read and predict
    :param model_name:
    :param image_data:
    :param input_op_name:
    :param predict_op_name:
    :return:
    """
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(name=model_name, mode='rb') as model_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(model_file.read())
            _ = tf.import_graph_def(graph_def, name='')
        for index, layer in enumerate(graph_def.node):
            print(index, layer.name)

    with tf.Session() as sess:
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        sess.run(init_op)
        image = image_data.eval()
        input = sess.graph.get_tensor_by_name(name=input_op_name)
        output = sess.graph.get_tensor_by_name(name=predict_op_name)

        predict_softmax = sess.run(fetches=output, feed_dict={input: image})
        predict_label = np.argmax(predict_softmax, axis=1)
        return predict_label


if __name__ == "__main__":

    train_num_samples = get_num_samples(record_dir=FLAGS.train_data)
    val_num_samples = get_num_samples(record_dir=FLAGS.val_data)
    # approximate samples per epoch

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(train_num_samples / FLAGS.batch_size))
    val_batches_per_epoch = int(np.floor(val_num_samples / FLAGS.batch_size))

    alex_net = AlexNet(input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                       num_classes=FLAGS.num_classes,
                       batch_size=FLAGS.batch_size,
                       learning_rate = FLAGS.learning_rate,
                       keep_prob=FLAGS.keep_prop,
                       weight_decay=FLAGS.weight_decay,
                       is_pretrain=FLAGS.is_pretrain)

    train_images, train_labels, train_filenames = dataset_tfrecord(record_file=FLAGS.train_data,
                                                                   batch_size=FLAGS.batch_size,
                                                                   input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                                   class_depth=FLAGS.num_classes,
                                                                   epoch=FLAGS.epoch,
                                                                   shuffle=True,
                                                                   is_training=True)

    val_images, val_labels, val_filenames = dataset_tfrecord(record_file=FLAGS.val_data,
                                                            batch_size=FLAGS.batch_size,
                                                            input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                            class_depth=FLAGS.num_classes,
                                                            epoch=FLAGS.epoch,
                                                            shuffle=True,
                                                            is_training=False)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    sess = tf.Session()
    with sess.as_default():
        sess.run(init_op)
        # get computer graph
        graph = tf.get_default_graph()
        write = tf.summary.FileWriter(logdir=FLAGS.logs_dir, graph=graph)
        # get model variable of network
        model_variable = tf.model_variables()
        for var in model_variable:
            print(var.name)
        # get and add histogram to summary protocol buffer
        logit_weight = graph.get_tensor_by_name(name='alexnet/fc8/weights:0')
        tf.summary.histogram(name='logits/weight', values=logit_weight)
        logit_biases = graph.get_tensor_by_name(name='alexnet/fc8/biases:0')
        tf.summary.histogram(name='logits/biases', values=logit_biases)
        # merges all summaries collected in the default graph
        summary_op = tf.summary.merge_all()
        # load pretrain model
        if FLAGS.is_pretrain:
            # remove variable of fc8 layer from pretrain model
            alex_net.load_pretrain_model(sess, FLAGS.pretrain_model_dir)

        # print(sess.run('vgg_16/conv1/conv1_1/biases:0'))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            if not coord.should_stop():
                # used to count the step per epoch

                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))

                    for train_step in range(train_batches_per_epoch):

                        train_image, train_label, train_filename = sess.run([train_images, train_labels, train_filenames])

                        feed_dict = alex_net.fill_feed_dict(image_feed=train_image,
                                                            label_feed=train_label,
                                                            is_training=True)

                        _, loss_value, train_accuracy, summary = sess.run(fetches=[alex_net.train, alex_net.loss,
                                                                                   alex_net.accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                        print('step {0}: train loss value {1}  train accuracy {2}'.
                              format(train_step, loss_value, train_accuracy))
                        write.add_summary(summary=summary, global_step=train_step)


                    val_acc = 0
                    for val_step in range(val_batches_per_epoch):
                        val_image, val_label, val_filename = sess.run(
                            [val_images, val_labels, val_filenames])

                        feed_dict = alex_net.fill_feed_dict(image_feed=val_image,
                                                            label_feed=val_label,
                                                            is_training=False)
                        val_acc += sess.run(fetches=alex_net.accuracy,
                                            feed_dict=feed_dict)

                    val_acc = val_acc / val_batches_per_epoch


                    print('epoch{0}: val accuracy value {1}'.format(epoch, val_acc))

                write.close()

                # save model
                # get op name for save model
                input_op = alex_net.raw_input_data.name
                logit_op = alex_net.logits.name
                # convert variable to constant
                input_graph_def = tf.get_default_graph().as_graph_def()
                constant_graph = tf.graph_util.convert_variables_to_constants(sess, input_graph_def,
                                                                              [input_op.split(':')[0],
                                                                               logit_op.split(':')[0]])
                # save to serialize file
                with tf.gfile.FastGFile(name=model_name, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')