# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/27 11:03

import tensorflow as tf


def filter_outside_boxes(anchors, img_h, img_w):
    '''
    :param anchors: list [-1,4], all levels' anchors
    :param img_h:  int type, image's height
    :param img_w:  int type,image's width
    :return: valid_indices: list [-1]
    '''
    with tf.name_scope('filter_outside_box'):
        y_min, x_min, y_max, x_max = tf.unstack(anchors, axis=1)
        y_min_index = tf.greater_equal(y_min, 0)
        x_min_index = tf.greater_equal(x_min, 0)
        y_max_index = tf.less(y_max, tf.cast(img_h, dtype=tf.float32))
        x_max_index = tf.less(x_max, tf.cast(img_w, dtype=tf.float32))

        indices = tf.transpose(tf.stack([y_min_index, x_min_index, y_max_index, x_max_index]))
        indices = tf.cast(indices, tf.float32)
        indices = tf.reduce_sum(indices, axis=1)
        valid_indices = tf.reshape(tf.where(tf.equal(indices, 4)), shape=[-1, ])

    return valid_indices


if __name__ == '__main__':
    anchors_min = tf.random_normal(shape=[6, 2], mean=1)
    anchors_max = anchors_min+1
    anchors = tf.concat([anchors_min, anchors_max], axis=1)
    img_h = 2
    img_w = 2
    valid_indices = filter_outside_box(anchors, img_h, img_w)

    with tf.Session() as sess:
        a, b = sess.run([anchors, valid_indices])
        print(a)
        print(b)

