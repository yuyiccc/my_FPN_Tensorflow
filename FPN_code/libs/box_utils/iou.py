# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/4 15:57
import tensorflow as tf


def calculate_iou(box1, box2):
    '''
    :param box1:[n,4] ->(ymin, xmin, ymax, xmax)
    :param box2: [m,4] ->(ymin, xmin, ymax, xmax)
    :return:  iou:[n,m]
    '''
    ymin_1, xmin_1, ymax_1, xmax_1 = tf.split(box1, num_or_size_splits=4, axis=1)  # ymin_1's shape is [n, 1]
    ymin_2, xmin_2, ymax_2, xmax_2 = tf.unstack(box2, axis=1)  # ymin_2's shape is [m,]

    ymin = tf.maximum(ymin_1, ymin_2)
    xmin = tf.maximum(xmin_1, xmin_2)
    ymax = tf.minimum(ymax_1, ymax_2)
    xmax = tf.minimum(xmax_1, xmax_2)

    overlap_h = tf.maximum(ymax - ymin, 0.0)
    overlap_w = tf.maximum(xmax - xmin, 0.0)

    overlap_area = overlap_h * overlap_w

    area_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)  # shape [n,1]
    area_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)  # shape [m,1]

    iou = overlap_area / (area_1 + area_2 - overlap_area)

    return iou


if __name__ == '__main__':
    box1 = tf.constant([[0, 0, 50, 50]], dtype=tf.float32)
    box2 = tf.constant([[0, 0, 50, 50], [0, 0, 50, 50], [50, 50, 100, 100], [200, 234, 547, 645]], dtype=tf.float32)

    iou = calculate_iou(box1, box2)
    matchs = tf.cast(tf.argmax(iou, axis=1), tf.int32)

    with tf.Session() as sess:
        iou_a = sess.run([iou, matchs])
        print(iou_a)
