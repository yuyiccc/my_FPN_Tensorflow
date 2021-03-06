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


def clip_boxes_to_img_boundaries(boxes, img_shape):
    '''
    :param boxes: [-1, 4]  ->[ymin, xmin, ymax, xmax]
    :param img_shape: [1,4]->[1,h,w,3]
    :return:
    '''
    def clip_one_side(one_side, max_boundaries):
        return tf.maximum(tf.minimum(one_side, max_boundaries), 0.0)

    image_h, image_w = tf.cast(img_shape[1], dtype=tf.float32), tf.cast(img_shape[2], dtype=tf.float32)

    ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
    ymin = clip_one_side(ymin, image_h)
    xmin = clip_one_side(xmin, image_w)
    ymax = clip_one_side(ymax, image_h)
    xmax = clip_one_side(xmax, image_w)

    return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def non_maximal_suppression(boxes, scores, iou_threshold, max_output_size, name='nms'):
    with tf.variable_scope(name):
        nms_indices = tf.image.non_max_suppression(boxes=boxes,
                                                   scores=scores,
                                                   max_output_size=max_output_size,
                                                   iou_threshold=iou_threshold,
                                                   name=name)
        return nms_indices


if __name__ == '__main__':
    anchors_min = tf.random_normal(shape=[6, 2], mean=1)
    anchors_max = anchors_min+1
    anchors = tf.concat([anchors_min, anchors_max], axis=1)
    img_h = 2
    img_w = 2
    valid_indices = filter_outside_boxes(anchors, img_h, img_w)
    image_shape = tf.constant([1, img_h, img_w, 3], dtype=tf.int32)
    clip_boxes = clip_boxes_to_img_boundaries(anchors, image_shape)

    with tf.Session() as sess:
        a, b, c = sess.run([anchors, valid_indices, clip_boxes])
        print(a)
        print(b)
        print(c)
