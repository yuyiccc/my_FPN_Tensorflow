# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 9:46
import os
import tensorflow as tf
import sys
sys.path.append('../../')
import configs.global_cfg as cfg
from data.process_image import random_flip_left_right, short_side_resize


class Read_tfrecord():
    def __init__(self, data_set_name=cfg.DATASET_NAME, batch_size=cfg.BATCH_SIZE, epoch=cfg.EPOCH, shortside_len=cfg.SHORTSIDE_LEN, is_training=True):
        if is_training:
            self.tfrecord_path = os.path.join(cfg.OUTER_PATH, 'tfrecords', data_set_name, 'train.tfrecord')
        else:
            self.tfrecord_path = os.path.join(cfg.OUTER_PATH, 'tfrecords', cfg.data_set_name, 'test*')
        self.batch = batch_size
        self.epoch = epoch
        self.shortside_len = shortside_len
        self.is_training = is_training

    def read_and_decode_single_example(self, example_proto):
        features = {
                'img_name': tf.FixedLenFeature([], tf.string),
                'img_height': tf.FixedLenFeature([], tf.int64),
                'img_width': tf.FixedLenFeature([], tf.int64),
                'img': tf.FixedLenFeature([], tf.string),
                'gtbox_label': tf.FixedLenFeature([], tf.string),
                'num_gtbox': tf.FixedLenFeature([], tf.int64)
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        img_name = parsed_features['img_name']
        img_height = tf.cast(parsed_features['img_height'], tf.int32)
        img_width = tf.cast(parsed_features['img_width'], tf.int32)
        img = tf.decode_raw(parsed_features['img'], tf.uint8)
        img = tf.reshape(img, shape=[img_height, img_width,3])
        gtboxes_label = tf.decode_raw(parsed_features['gtbox_label'], tf.int32)
        gtboxes_label = tf.reshape(gtboxes_label, shape=[-1, 5])
        num_gtbox = tf.cast(parsed_features['num_gtbox'], tf.int32)
        return img_name, img, gtboxes_label, num_gtbox

    def read_and_process_single_data(self, example_proto):
        img_name, img, gtboxes_label, num_gtbox = self.read_and_decode_single_example(example_proto)

        # uint8->float32
        img = tf.cast(img, tf.float32)
        # minus mean value
        img -= tf.constant(cfg.DEPTH_MEAN)
        # resize
        img, gtboxes_label = short_side_resize(img_tensor=img, gtbox_tensor=gtboxes_label, shortside_len=self.shortside_len)
        # flip for data augmentation
        if self.is_training:
            img, gtboxes_label = random_flip_left_right(img_tensor=img, gtbox_tensor=gtboxes_label)

        return img, img_name, gtboxes_label, num_gtbox

    def get_batch_data(self):
        print("tfrecord's path is :%s"%self.tfrecord_path)
        data_set = tf.data.TFRecordDataset(self.tfrecord_path, num_parallel_reads=16)
        data_set = data_set.map(self.read_and_process_single_data, num_parallel_calls=16)
        data_set = data_set.repeat(self.epoch)
        data_set = data_set.batch(self.batch)
        data_set = data_set.prefetch(self.batch*2)
        iterator = data_set.make_initializable_iterator()
        img, img_name, gtboxes_label, num_gtbox = iterator.get_next()
        return iterator, img_name, img, gtboxes_label, num_gtbox

# test it
if __name__=='__main__':
    data = Read_tfrecord()
    iterator, img_name, img, gtboxes_label, num_gtbox = data.get_batch_data()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10):
            img_name_i, img_i, gtboxes_label_i, num_gtbox_i = sess.run((img_name, img, gtboxes_label, num_gtbox))
            print(img_name_i[0].decode(), i)
            print(img_i.shape)
            print(gtboxes_label_i, num_gtbox_i)
            print(img_i[:4, :4, 0], '\n\n')

