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
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
# from VGG16.VGG16 import VGG16
from AlexNet_Tensorflow.nets.AlexNet_slim import AlexNet
import numpy as np
from DataProcess.read_TFRecord import dataset_tfrecord, get_num_samples


# dataset path
dataset_dir = '/home/alex/Documents/dataset/flower_tfrecord'
train_data = os.path.join(dataset_dir, 'train')
val_data = os.path.join(dataset_dir, 'val')

# model_path
pretrain_model_dir = '/home/alex/Documents/pretrain_model/alexnet/alexnet.npy'
model_dir = os.path.join('../', 'outputs', 'model')
logs_dir = os.path.join('../', 'outputs', 'logs')


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('height', 227, 'Number of height size.')
flags.DEFINE_integer('width', 227, 'Number of width size.')
flags.DEFINE_integer('depth', 3, 'Number of depth size.')
flags.DEFINE_integer('num_classes', 5, 'Number of image class.')
flags.DEFINE_integer('batch_size', 32, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('epoch', 30, 'Number of epoch size.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('keep_prop', 0.8, 'Number of probability that each element is kept.')
flags.DEFINE_float('weight_decay', 0.0005, 'Number of regular scale size')
flags.DEFINE_bool('is_pretrain', True, 'if True, use pretrain model.')
flags.DEFINE_string('pretrain_model_dir', pretrain_model_dir, 'Directory to restore pretrain model dir.')
flags.DEFINE_string('train_data', train_data, 'Directory to put the training data.')
flags.DEFINE_string('val_data', val_data, 'Directory to put the validation data.')
flags.DEFINE_string('model_dir',model_dir, 'Directory to save model.')
flags.DEFINE_string('logs_dir', logs_dir, 'direct of summary logs.')
flags.DEFINE_integer('save_step_period', 2000, 'save model step period')


if __name__ == "__main__":

    num_train_samples = get_num_samples(record_dir=FLAGS.train_data)
    num_val_samples = get_num_samples(record_dir=FLAGS.val_data)
    # approximate samples per epoch

    # get total step of the number train epoch
    train_step_per_epoch = num_train_samples // FLAGS.batch_size  # get num step of per epoch
    # max_step = FLAGS.epoch * step_per_epoch  # get total step of several epoch

    # get the number step of one validation epoch
    val_step_per_epoch = num_val_samples // FLAGS.batch_size

    alex_net = AlexNet(input_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                       num_classes=FLAGS.num_classes,
                       batch_size=FLAGS.batch_size,
                       learning_rate = FLAGS.learning_rate,
                       keep_prob=FLAGS.keep_prop,
                       weight_decay=FLAGS.weight_decay)

    train_images_batch, train_labels_batch, train_filenames = dataset_tfrecord(record_files=FLAGS.train_data,
                                                                   batch_size=FLAGS.batch_size,
                                                                   target_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                                   class_depth=FLAGS.num_classes,
                                                                   epoch=FLAGS.epoch,
                                                                   shuffle=True,
                                                                   is_training=True)

    val_images_batch, val_labels_batch, val_filenames = dataset_tfrecord(record_files=FLAGS.val_data,
                                                            batch_size=FLAGS.batch_size,
                                                            target_shape=[FLAGS.height, FLAGS.width, FLAGS.depth],
                                                            class_depth=FLAGS.num_classes,
                                                            epoch=FLAGS.epoch,
                                                            shuffle=True,
                                                            is_training=False)
    saver = tf.train.Saver()
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    # train and save model
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
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
        tf.summary.histogram(name='logits/weights', values=logit_weight)
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
                train_step = 0
                for epoch in range(FLAGS.epoch):
                    print('Epoch: {0}/{1}'.format(epoch, FLAGS.epoch))

                    # ++++++++++++++++++++++++++++++++++++++++train part+++++++++++++++++++++++++++++++++++++++++++++
                    for step in range(train_step_per_epoch):

                        train_image, train_label, train_filename = sess.run([train_images_batch, train_labels_batch, train_filenames])

                        feed_dict = alex_net.fill_feed_dict(image_feed=train_image,
                                                            label_feed=train_label,
                                                            is_training=True)

                        _, loss_value, train_accuracy, summary = sess.run(fetches=[alex_net.train, alex_net.loss,
                                                                                   alex_net.accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                        print('\tstep {0}: train loss value {1}  train accuracy {2}'.
                              format(step, loss_value, train_accuracy))
                        write.add_summary(summary=summary, global_step=train_step)
                    # ++++++++++++++++++++++++++++++++++++++++validation part+++++++++++++++++++++++++++++++++++++
                    val_losses = []
                    val_accuracies = []
                    for _ in range(val_step_per_epoch):
                        val_images, val_labels = sess.run([val_images_batch, val_labels_batch])

                        feed_dict = alex_net.fill_feed_dict(image_feed=val_images, label_feed=val_labels,
                                                            is_training=False)
                        val_loss, val_acc = sess.run([alex_net.loss, alex_net.accuracy], feed_dict=feed_dict)

                        val_losses.append(val_loss)
                        val_accuracies.append(val_acc)
                    mean_loss = np.array(val_losses, dtype=np.float32).mean()
                    mean_acc = np.array(val_accuracies, dtype=np.float32).mean()

                    print("\t{0}: epoch: {1}  val Loss: {2}, val accuracy:  {3}".format(datetime.now(),
                                                                                        epoch,
                                                                                        mean_loss, mean_acc))

                    train_step += 1 # update train step
                    if train_step % FLAGS.save_step_period == 0:
                        saver.save(sess, save_path=os.path.join(FLAGS.model_dir, 'model.ckpt'),
                                   global_step=alex_net.global_step)

                saver.save(sess, save_path=os.path.join(FLAGS.model_dir, 'model.ckpt'),
                           global_step=alex_net.global_step)
            write.close()
        except Exception as e:
            print(e)
        coord.request_stop()
        coord.join(threads)
    sess.close()
    print('model training has complete')
