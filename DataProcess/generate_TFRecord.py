#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : generate_TFRecord.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2019/12/20 PM 16:05
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from Utils.tools import view_bar, makedir

original_dataset_dir = '/home/alex/Documents/dataset/flower_split'
train_src = os.path.join(original_dataset_dir, 'train')
val_src = os.path.join(original_dataset_dir, 'val')

target_dataset_dir = '/home/alex/Documents/dataset/flower_tfrecord'

train_target = os.path.join(target_dataset_dir, 'train')
val_target = os.path.join(target_dataset_dir, 'val')


#+++++++++++++++++++++++++++++++++++++++++generate tfrecord+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def execute_tfrecord(source_path, outputs_path, per_record_capacity=500, shuffle=True):
    """

    :param source_path:
    :param outputs_path:
    :param split_ratio:
    :param per_record_capacity:
    :param shuffle:
    :return:
    """

    img_names, img_labels, classes_map = get_label_data(source_path, shuffle=shuffle)

    # test_record_path = os.path.join(outputs_path, 'test')
    makedir(outputs_path)

    # save class map infor
    with open(os.path.join(outputs_path, 'label_map.txt'), 'w') as fw:
        for index, name in classes_map.items():
            fw.write(f'{index},{name}\n')

    num_samples = len(img_names)
    # test_data_num = int(num_samples * split_ratio)
    # train_data_num = num_samples - test_data_num
    #
    # train_name_list = img_names[:train_data_num]
    # train_labels_list = img_labels[:train_data_num]
    # test_name_list = img_names[train_data_num:]
    # test_labels_list = img_labels[train_data_num:]

    image_to_record(save_path=outputs_path,
                    img_name_list=img_names,
                    labels_list=img_labels,
                    record_capacity=per_record_capacity)
    print("\nThere are {0} samples has successfully convert to tfrecord, save at {1}".format(num_samples,
                                                                                           outputs_path))
    # image_to_record(save_path=test_record_path,
    #                 img_name_list=test_name_list,
    #                 labels_list=test_labels_list,
    #                 record_capacity=per_record_capacity)
    # print("There are {0} samples has successfully convert to tfrecord, save at {1}".format(test_data_num,
    #                                                                                        test_record_path))


def image_to_record(save_path, img_name_list, labels_list=None, record_capacity=500):
    """

    :param save_path:
    :param img_name_list: Str
    :param img_list: Array, only used for binary format
    :param format:
    :param img_height: only used for binary format
    :param img_width: only used for binary format
    :param img_depth: only used for binary format
    :param label_map:
    :param record_capacity:
    :return:
    """
    import cv2 as cv

    remainder_num = len(img_name_list) % record_capacity
    if remainder_num == 0:
        num_record = int(len(img_name_list) / record_capacity)
    else:
        num_record = int(len(img_name_list) / record_capacity) + 1

    count = 1
    for index in range(num_record):
        record_filename = os.path.join(save_path, 'tfrecord-{0}.record'.format(index))
        writer = tf.io.TFRecordWriter(record_filename)
        if index < num_record - 1:
            sub_img_name_list = img_name_list[index * record_capacity: (index + 1) * record_capacity]
            sub_label_list = labels_list[index * record_capacity: (index + 1) * record_capacity]
        else:
            sub_img_name_list = img_name_list[(index * record_capacity): (index * record_capacity + remainder_num)]
            sub_label_list = labels_list[(index * record_capacity): (index * record_capacity + remainder_num)]

        for img_name, label in zip(sub_img_name_list, sub_label_list):

            image_bgr = cv.imread(img_name)
            image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            height = image.shape[0]
            width = image.shape[1]
            depth = image.shape[2]

            image_record = image_example(image=image, label=label, img_height=height, img_width=width, img_depth=depth,
                                         filename=img_name)
            writer.write(record=image_record)

            view_bar(message='Conversion progress', num=count, total=len(img_name_list))
            count += 1

        writer.close()

    return True

def get_label_data(data_path, classes_map=None, shuffle=True):
    """
    get image list and label list
    :param data_path:
    :return:
    """

    img_names = []  # image name
    img_labels = []  # image label
    class_map = {}

    if classes_map is None:
        # classes name
        # sorted operation ensure to the train and validation dataset has equal class map
        for subdir in sorted(os.listdir(data_path)):
            if os.path.isdir(os.path.join(data_path, subdir)):
                class_map[subdir] = len(class_map)
    else:
        class_map = classes_map


    for class_name, class_label in class_map.items():
        # get image file each of class
        class_dir = os.path.join(data_path, class_name)
        image_list = os.listdir(class_dir)

        for img_name in image_list:
            img_names.append(os.path.join(class_dir, img_name))
            img_labels.append(class_label)

    num_samples = len(img_names)

    if shuffle:
        img_names_shuffle = []
        img_labels_shuffle = []
        index_array = np.random.permutation(num_samples)

        for i, index in enumerate(index_array):
            img_names_shuffle.append(img_names[index])
            img_labels_shuffle.append(img_labels[index])

        img_names = img_names_shuffle
        img_labels = img_labels_shuffle


    return img_names, img_labels, class_map


# protocol buffer(protobuf)
# Example 是 protobuf 协议下的消息体
# 一个Example 消息体包含一系列 feature 属性
# 每个feature 是一个map (key-value)
# key 是 String类型：
# value 是 Feature 类型的消息体，取值有三种类型： BytesList， FloatList， Int64List

def _bytes_feature(value):
    """
    return bytes_list from a string / bytes
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    return float_list from a float / double
    :param value:
    :return:
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    return an int64_list from a bool/enum/int/uint.
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image, label, img_height, img_width, img_depth, filename):
    """
    create a tf.Example message to be written to a file
    :param label: label info
    :param image: image content
    :param filename: image name
    :return:
    """

    # create a dict mapping the feature name to the tf.Example compatible
    # image_shape = tf.image.decode_jpeg(image_string).eval().shape
    feature = {
        "image": _bytes_feature(image.tostring()),
        "label": _int64_feature(label),
        "height": _int64_feature(img_height),
        "width": _int64_feature(img_width),
        "depth": _int64_feature(img_depth),
        "filename": _bytes_feature(filename.encode())
    }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def serialize_example(label, image, filename):
    """
    create a tf.Example message to be written to a file
    :param label: label info
    :param image: image content
    :param filename: image name
    :return:
    """
    # create a dict mapping the feature name to the tf.Example compatible
    feature = {
        "label": _int64_feature(label),
        "image": _bytes_feature(image),
        "filename": _bytes_feature(filename)
    }
    # create a feature message using tf.train.Example
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(label, image, filename):
    tf_string = tf.py_function(func=serialize_example,
                               inp=(label, image, filename),
                               Tout=tf.string)
    # the result is scalar
    return tf.reshape(tf_string, ())


def decode_message(message):
    """
    decode message from string
    :param message:
    :return:
    """
    return tf.train.Example.FromString(message)


#++++++++++++++++++++++++++++++++++++++++++tfrecord test++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def tfrecord_test():
    """
    test tfrecord
    :return:
    """
    # test feature type
    f0 = _bytes_feature([b'alex'])  # <class 'tensorflow.core.example.feature_pb2.Feature'>
    f1 = _bytes_feature([u'alex'.encode('utf8')])
    f2 = _float_feature([np.exp(1)])
    f3 = _int64_feature([True])
    print(type(f0))
    # serialize to binary-string
    fs = f3.SerializeToString()
    print(fs)

    # create observation dataset
    n_observation = int(1e4)
    # boolean feature
    f4 = np.random.choice([True, False], n_observation)
    # integer feature
    f5 = np.random.randint(0, 5, n_observation)
    class_str = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    # string(byte) feature
    f6 = class_str[f5]
    # string(byte) feature
    f7 = np.random.choice([b'yes', b'no'], n_observation)

    example_observation = serialize_example([True], [b'alex'], [b'yes'])
    print(example_observation)
    print(decode_message(example_observation))

    # return dataset of scalar
    feature_dataset = tf.data.Dataset.from_tensor_slices((f5, f6, f7))
    print(feature_dataset)
    # print(tf_serialize_example(f5, f6, f7))
    # apply the function to each element in dataset
    serialize_feature_dataset = feature_dataset.map(tf_serialize_example)
    print(serialize_feature_dataset)

    data_path = '/home/alex/Documents/datasets/dogs_and_cat_separate'

    # img_shape = tf.image.decode_jpeg(byte_img)
    # print(image_example(1, byte_img, b'cat'))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++main+++++++++++++++++++++++++++++++++++++++++++++++++++++++++


if __name__ == "__main__":

    execute_tfrecord(source_path=train_src, outputs_path=train_target)
    execute_tfrecord(source_path=val_src, outputs_path=val_target)

