# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/3 10:02

import tensorflow as tf


def decode_boxes(encode_boxes, reference_boxes, scale_factors=None, name='decode_boxes'):
    '''
    :param encode_boxes: [n, 4]
    :param reference_boxes: [n, 4]
    :param scale_factors: [1, 4]
    :param name:  ops name
    :return: all_decode_boxes: [n, 4]
    '''
    with tf.variable_scope(name):
        t_ycenter, t_xcenter, t_h, t_w = tf.unstack(encode_boxes, axis=1)
        if scale_factors:
            t_ycenter /= scale_factors[0]
            t_xcenter /= scale_factors[1]
            t_h /= scale_factors[2]
            t_w /= scale_factors[3]

        refe_ymin, refe_xmin, refe_ymax, refe_xmax = tf.unstack(reference_boxes, axis=1)
        refe_ycenter = (refe_ymax + refe_ymin) / 2
        refe_xcenter = (refe_xmax + refe_xmin) / 2
        refe_h = refe_ymax - refe_ymin
        refe_w = refe_xmax - refe_xmin

        predict_ycenter = refe_h * t_ycenter + refe_ycenter
        predict_xcenter = refe_w * t_xcenter + refe_xcenter
        predict_w = refe_w * tf.exp(t_w)
        predict_h = refe_h * tf.exp(t_h)

        predict_xmin = predict_xcenter - predict_w / 2
        predict_ymin = predict_ycenter - predict_h / 2
        predict_xmax = predict_xcenter + predict_w / 2
        predict_ymax = predict_ycenter + predict_h / 2

        all_decode_boxes = tf.transpose(tf.stack([predict_ymin, predict_xmin, predict_ymax, predict_xmax]))

        return all_decode_boxes


def encode_boxes(anchors, gtboxes, scale_factors=None):
    with tf.variable_scope('encode_boxes'):
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = tf.unstack(gtboxes, axis=1)
        gt_xcenter, gt_ycenter = (gt_xmax + gt_xmin)/2, (gt_ymax + gt_ymin)/2
        gt_h, gt_w = (gt_ymax - gt_ymin), (gt_xmax - gt_xmin)

        anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax  = tf.unstack(anchors, axis=1)
        anchor_ycenter, anchor_xcenter = (anchor_ymax + anchor_ymin)/2, (anchor_xmax + anchor_xmin)/2
        anchor_h, anchor_w = (anchor_ymax - anchor_ymin), (anchor_xmax - anchor_xmin)

        # there may some NaN problem in tf.log if gt_h=0 or gt_w=0
        # and if  anchor_h=0 or anchor_w=0, but it should not be zeros
        encode_boxes_h, encode_boxes_w = tf.log(gt_h/anchor_h), tf.log(gt_w/anchor_w)
        encode_boxes_ycenter = (gt_ycenter-anchor_ycenter)/anchor_h
        encode_boxes_xcenter = (gt_xcenter-anchor_xcenter)/anchor_w

        if scale_factors:
            encode_boxes_ycenter *= scale_factors[0]
            encode_boxes_xcenter *= scale_factors[1]
            encode_boxes_h *= scale_factors[2]
            encode_boxes_w *= scale_factors[3]
        all_encode_boxes = tf.transpose(tf.stack([encode_boxes_ycenter,
                                                  encode_boxes_xcenter,
                                                  encode_boxes_h,
                                                  encode_boxes_w]
                                                 )
                                        )
    return all_encode_boxes


if __name__ == '__main__':
    encode_boxes_ = tf.constant([[1, 1, -1, -2], [2, 3, -4, -5]], dtype=tf.float32)
    reference_boxes = tf.constant([[0, 0, 50, 50], [0, 0, 50, 50]], dtype=tf.float32)
    all_decode_boxes = decode_boxes(encode_boxes_, reference_boxes)

    encode_all_decode_boxes = encode_boxes(reference_boxes, all_decode_boxes)

    ini = tf.group(tf.local_variables_initializer(),
                   tf.global_variables_initializer())
    with tf.Session() as sess:
        sess.run(ini)
        boxes_a, boxes_b = sess.run([all_decode_boxes, encode_all_decode_boxes])
        print(boxes_a)
        print(boxes_b)
