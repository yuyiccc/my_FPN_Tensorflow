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


if __name__ == '__main__':
    encode_boxes = tf.constant([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=tf.float32)
    reference_boxes = tf.constant([[0, 0, 50, 50], [0, 0, 50, 50]], dtype=tf.float32)
    all_decode_boxes = decode_boxes(encode_boxes, reference_boxes)

    ini = tf.group(tf.local_variables_initializer(),
                   tf.global_variables_initializer())
    with tf.Session() as sess:
        sess.run(ini)
        boxes = sess.run(all_decode_boxes)
        print(boxes)
