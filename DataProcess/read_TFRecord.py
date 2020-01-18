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

# original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
original_dataset_dir = '/home/alex/Documents/datasets/dogs_vs_cat_separate'
tfrecord_dir = os.path.join(original_dataset_dir, 'tfrecord')

train_path = os.path.join(original_dataset_dir, 'train')
test_path = os.path.join(original_dataset_dir, 'test')


def parse_example(serialized_sample, input_shape, class_depth):
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
    image = augmentation_image(input_image=image, image_shape=input_shape)
    # onehot label
    label = tf.one_hot(indices=label, depth=class_depth)

    return image, label, filename


def augmentation_image(input_image, image_shape, flip_lr=False, flit_ud=False, brightness=False,
                       bright_delta=0.2, contrast=False, contrast_lower=0.5, contrast_up=1.5, hue=False,
                       hue_delta=0.2, saturation=False, saturation_low=0.5, saturation_up=1.5, standard=False):
    # enlarge image to same size
    resize_img = tf.image.resize(images=input_image, size=(int(1.2*image_shape[0]), int(1.2*image_shape[1])))

    try:
        # crop image
        distort_img = tf.image.random_crop(value=tf.cast(resize_img, tf.uint8), size=image_shape, seed=0)
        # flip image in left and right
        if flip_lr:
            distort_img = tf.image.random_flip_left_right(image=distort_img, seed=0)
        # flip image in left and right
        if flit_ud:
            distort_img = tf.image.random_flip_up_down(image=distort_img, seed=0)
        # adjust image brightness
        if brightness:
            distort_img = tf.image.random_brightness(image=distort_img, max_delta=bright_delta)
        # # adjust image contrast
        if contrast:
            distort_img = tf.image.random_contrast(image=distort_img, lower=contrast_lower, upper=contrast_up)
        # adjust image hue
        if hue:
            distort_img = tf.image.random_hue(image=distort_img, max_delta=hue_delta)
        #  adjust image saturation
        if saturation:
            distort_img = tf.image.random_saturation(image=distort_img, lower=saturation_low, upper=saturation_up)
        # reduce pixel value to (0., 1.)
        # normalize_img = distort_img / 255.
        # image standard process
        # if standard:
        #     distort_img = tf.image.per_image_standardization(image=distort_img)
            # resize image
        distort_img = tf.image.resize_images(images=distort_img, size=image_shape[:-1])
        return distort_img
    except Exception as e:
        print(e)


def dataset_tfrecord(record_file, input_shape, class_depth, epoch=5, batch_size=10, shuffle=True):
    """
    construct iterator to read image
    :param record_file:
    :return:
    """
    record_list = []
    # check record file format
    if os.path.isfile(record_file):
        record_list = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_list.append(os.path.join(record_file, filename))
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
                                            parse_example(series_record, input_shape, class_depth))
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


def reader_tfrecord(record_file, input_shape, class_depth, batch_size=10, num_threads=2, epoch=5, shuffle=True):
    """
    read and sparse TFRecord
    :param record_file:
    :return:
    """
    record_tensor = []
    # check record file format
    if os.path.isfile(record_file):
        record_tensor = [record_file]
    else:
        for filename in os.listdir(record_file):
            record_tensor.append(os.path.join(record_file, filename))
    # create input queue
    filename_queue = tf.train.string_input_producer(string_tensor=record_tensor, num_epochs=epoch, shuffle=shuffle)
    # create reader to read TFRecord sample instant
    reader = tf.TFRecordReader()
    # read one sample instant
    _, serialized_sample = reader.read(filename_queue)

    # parse sample
    image, label, filename = parse_example(serialized_sample, input_shape=input_shape, class_depth=class_depth)

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

    record_list = []
    # check record file format

    for filename in os.listdir(record_dir):
        record_list.append(os.path.join(record_dir, filename))

    num_samples = 0
    for record_file in record_list:
        for record in tf_record_iterator(record_file):
            num_samples += 1
    return num_samples

if __name__ == "__main__":
    record_file = os.path.join(tfrecord_dir, 'train')
    image_batch, label_batch, filename = dataset_tfrecord(record_file=record_file, input_shape=[224, 224, 3],
                                                          class_depth=5)
    # create local and global variables initializer group
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    with tf.Session() as sess:
        sess.run(init_op)

        num_samples = get_num_samples(record_file)
        print('all sample size is {0}'.format(num_samples))
        # create Coordinator to manage the life period of multiple thread
        coord = tf.train.Coordinator()
        # Starts all queue runners collected in the graph to execute input queue operation
        # the step contain two operation:filename to filename queue and sample to sample queue
        threads = tf.train.start_queue_runners(coord=coord)
        print('threads: {0}'.format(threads))
        try:
            if not coord.should_stop():
                image_feed, label_feed = sess.run([image_batch, label_batch])
                plt.imshow(image_feed[0])
                plt.show()
        except Exception as e:
            print(e)
        finally:
            # request to stop all background threads
            coord.request_stop()

        # waiting all threads safely exit
        coord.join(threads)
        sess.close()

