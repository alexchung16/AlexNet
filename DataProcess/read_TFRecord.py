#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @ File generate_TFRecord.py
# @ Description :
# @ Author alexchung
# @ Time 17/10/2019 PM 15:48

import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tensorflow.python_io import tf_record_iterator
from DataProcess.alexnet_preprocessing import preprocess_image

dataset_dir = '/home/alex/Documents/dataset/flower_tfrecord'

train_data_path = os.path.join(dataset_dir, 'train')
test_data_path = os.path.join(dataset_dir, 'val')


def parse_example(serialized_sample, target_shape, class_depth, is_training=False):
    """
    parse tensor
    :param image_sample:
    :return:
    """

    # construct feature description
    image_feature_description ={

        "image": tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        "label": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "width": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "depth": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        "filename": tf.io.FixedLenFeature(shape=[], dtype=tf.string)
    }
    feature = tf.io.parse_single_example(serialized=serialized_sample, features=image_feature_description)

    # parse feature
    raw_img = tf.decode_raw(feature['image'], tf.uint8)
    # shape = tf.cast(feature['shape'], tf.int32)
    height = tf.cast(feature['height'], tf.int32)
    width = tf.cast(feature['width'], tf.int32)
    depth = tf.cast(feature['depth'], tf.int32)

    image = tf.reshape(raw_img, [height, width, depth])
    label = tf.cast(feature['label'], tf.int32)
    filename = tf.cast(feature['filename'], tf.string)
    # resize image shape
    # random crop image
    # before use shuffle_batch, use random_crop to make image shape to special size
    # first step enlarge image size
    # second step dataset operation

    # image augmentation
    # image = augmentation_image(image=image, image_shape=input_shape, preprocessing_type=preprocessing_type,
    #                            fast_mode=fast_mode, is_training=is_training,)
    image = preprocess_image(image=image, output_height=target_shape[0], output_width=target_shape[1],
                             is_training=is_training)
    # onehot label
    label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def dataset_tfrecord(record_files, target_shape, class_depth, epoch=5, batch_size=10, shuffle=True,
                    is_training=False):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    # check record file format
    if os.path.isfile(record_files):
        record_list = [record_files]
    else:
        record_list = [os.path.join(record_files, record_file) for record_file in os.listdir(record_files)
                       if record_file.split('.')[-1] == 'record']
    # # use dataset read record file
    raw_img_dataset = tf.data.TFRecordDataset(record_list)
    # execute parse function to get dataset
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements, in the
    # same order as they appeared in the input.
    # when parse_example has only one parameter (office recommend)
    # parse_img_dataset = raw_img_dataset.map(parse_example)
    # when parse_example has more than one parameter which used to process data
    parse_img_dataset = raw_img_dataset.map(lambda series_record:
                                            parse_example(series_record, target_shape, class_depth,
                                                          is_training=is_training))
    # get dataset batch
    if shuffle:
        shuffle_batch_dataset = parse_img_dataset.shuffle(buffer_size=batch_size*4).repeat(epoch).batch(batch_size=batch_size)
    else:
        shuffle_batch_dataset = parse_img_dataset.repeat(epoch).batch(batch_size=batch_size)
    # make dataset iterator
    image, label, filename = shuffle_batch_dataset.make_one_shot_iterator().get_next()

    # image = augmentation_image(input_image=image, image_shape=input_shape)
    # # onehot label
    # label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def reader_tfrecord(record_files, target_shape, class_depth, batch_size=10, num_threads=2, epoch=5, shuffle=True,
                    is_training=False):
    """
    read and sparse TFRecord
    :param record_file:
    :return:
    """
    # check record file format
    if os.path.isfile(record_files):
        record_list = [record_files]
    else:
        record_list = [os.path.join(record_files, record_file) for record_file in os.listdir(record_files)
                       if record_file.split('.')[-1] == 'record']
    # create input queue
    filename_queue = tf.train.string_input_producer(string_tensor=record_list, num_epochs=epoch, shuffle=shuffle)
    # create reader to read TFRecord sample instant
    reader = tf.TFRecordReader()
    # read one sample instant
    _, serialized_sample = reader.read(filename_queue)

    # parse sample
    image, label, filename = parse_example(serialized_sample, target_shape=target_shape, class_depth=class_depth,
                                           is_training=is_training)

    if shuffle:
        image, label, filename = tf.train.shuffle_batch([image, label, filename],
                                          batch_size=batch_size,
                                          capacity=batch_size * 4,
                                          num_threads=num_threads,
                                          min_after_dequeue=batch_size)
    else:
        image, label, filename = tf.train.batch([image, label, filename],
                                                batch_size=batch_size,
                                                capacity=batch_size,
                                                num_threads=num_threads,
                                                enqueue_many=False
                                                )
    # dataset = tf.data.Dataset.shuffle(buffer_size=batch_size*4)
    return image, label, filename


def get_num_samples(record_dir):
    """
    get tfrecord numbers
    :param record_file:
    :return:
    """

    record_list = [os.path.join(record_dir, record_file) for record_file in os.listdir(record_dir)
                   if record_file.split('.')[-1] == 'record']

    num_samples = 0
    for record_file in record_list:
        for _ in tf_record_iterator(record_file):
            num_samples += 1

    return num_samples

if __name__ == "__main__":
    num_samples = get_num_samples(train_data_path)
    print('all sample size is {0}'.format(num_samples))
    image_batch, label_batch, filename = dataset_tfrecord(record_files=train_data_path, target_shape=[227, 227, 3],
                                                          class_depth=5, is_training=True)

    # create local and global variables initializer group
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        print('threads: {0}'.format(threads))
        try:
            if not coord.should_stop():
                image_feed, label_feed = sess.run([image_batch, label_batch])
                plt.imshow(np.uint8(image_feed[0][:,:,::-1]))
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()

        # waiting all threads safely exit
        coord.join(threads)
        sess.close()

