# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/26 15:32

import tensorflow as tf

def make_anchors(base_anchor_size,
                 anchor_scales,
                 anchor_ratios,
                 feature_h,
                 feature_w,
                 stride,
                 name='make_anchors'):
    '''
    :param base_anchor_size:  int type, base anchor's size
    :param anchor_scales:  list type, anchor's  scales
    :param anchor_ratios:  list type, anchor's ratios
    :param feature_h:  int type, feature's height
    :param feature_w:  int type, feature's width
    :param stride:  int type, this feature map's stride to  original picture
    :param name:  string type, this ops name
    :return: anchors: [1,h,w,num_anchors_per_location,4]->[-1,4]
    '''
    with tf.variable_scope(name):
        # create different ratios and scales anchors
        # [num_anchors_per_location, 4]
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], dtype=tf.float32)
        base_anchors = enum_ratios(enum_scales(base_anchor, anchor_ratios), anchor_scales)

        _, _, hs, ws = tf.unstack(base_anchors, axis=1)

        x_centers = (tf.range(tf.cast(feature_w, dtype=tf.float32), dtype=tf.float32)+0.5)*stride
        y_centers = (tf.range(tf.cast(feature_h, dtype=tf.float32), dtype=tf.float32)+0.5)*stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        box_center = tf.reshape(tf.stack([y_centers, x_centers], axis=2), shape=[-1, 2])

        box_size = tf.reshape(tf.stack([hs, ws], axis=2), shape=[-1, 2])

        final_anchors = tf.concat([box_center-0.5*box_size, box_center+0.5*box_size], axis=1)
        return final_anchors



def enum_ratios(base_anchor, anchor_ratios, name='enum_ratios'):
    '''
    :param base_anchor:
    :param anchor_ratios: h/w
    :param name:
    :return:
    '''
    _, _, h, w = tf.unstack(base_anchor, axis=1)
    ratios = tf.expand_dims(anchor_ratios, axis=1)
    hs = tf.reshape(h*ratios, shape=[-1])
    ws = tf.reshape(w/ratios, shape=[-1])

    num_anchor_per_location = tf.shape(ws)[0]

    return tf.transpose(tf.stack([tf.zeros([num_anchor_per_location]),
                                  tf.zeros([num_anchor_per_location]),
                                  hs, ws]))

def enum_scales(base_anchor, anchor_scales, name='enum_scales'):
    '''
    :param base_anchor:  [y_center, x_center, h, w]
    :param anchor_scales: [0.5, 1, 2]
    :param name:
    :return: anchors
    '''
    with tf.variable_scope(name):
        anchor_scales = tf.reshape(anchor_scales, [-1, 1])
        return base_anchor*anchor_scales


if __name__=='__main__':
    anchors = make_anchors(base_anchor_size=16,
                           anchor_scales=[0.5, 1, 2],
                           anchor_ratios=[0.5, 1, 2],
                           feature_h=20,
                           feature_w=10,
                           stride=16,
                           name='try_this_part')
    with tf.Session() as sess:
        acc = sess.run(anchors)
        print(acc)
