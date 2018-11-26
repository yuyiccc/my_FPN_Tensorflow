# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/26 15:32

import tensorflow as tf
import numpy as np
import cv2

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
        base_anchors = enum_ratios(enum_scales(base_anchor, anchor_scales), anchor_ratios)

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
    ratios = tf.expand_dims(tf.sqrt(tf.cast(anchor_ratios, dtype=tf.float32)), axis=1)
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
        anchor_scales = tf.reshape(tf.cast(anchor_scales, dtype=tf.float32), shape=[-1, 1])
        return base_anchor*anchor_scales


def draw_anchor_boxes(anchors, img_h, img_w):
    img_backgrand = np.zeros([img_h*2, img_w*2, 3], dtype=np.uint8)
    color = (255, 0, 0)
    cv2.rectangle(img_backgrand,
                  pt1=(int(img_w*0.1), int(img_h*0.1)),
                  pt2=(int(img_w*1.1), int(img_h*1.1)),
                  color=(255, 255, 255),
                  thickness=2)

    for box in anchors:
        y_min, x_min, y_max, x_max = box
        cv2.rectangle(img_backgrand,
                      pt1=(int(img_w*0.1+x_min), int(img_h*0.1+y_min)),
                      pt2=(int(img_w*0.1+x_max), int(img_h*0.1+y_max)),
                      color=color,
                      thickness=1)
    cv2.imshow('show_anchors', img_backgrand)
    cv2.waitKey(0)

if __name__=='__main__':
    base_anchor_size = 128
    stride = 128
    feature_h = 5
    feature_w = 5

    anchors = make_anchors(base_anchor_size=base_anchor_size,
                           anchor_scales=[0.5, 0.8],
                           anchor_ratios=[0.5, 1, 2],
                           feature_h=feature_h,
                           feature_w=feature_w,
                           stride=stride,
                           name='try_this_part')
    with tf.Session() as sess:
        acc = sess.run(anchors)
        draw_anchor_boxes(acc, feature_h*stride, feature_w*stride)
        print(acc)
