# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 14:52
import tensorflow as tf
import  numpy as np

def short_side_resize(img_tensor, gtbox_tensor, shortside_len):
    '''
    :param img_tensor: [h,w,c]
    :param gtbox_tensor: [-1,5]
    :param shortside_len: 600 (set in /configs/global_cfg.py)
    :return: resized img_tenor and gtbox_tensor which have the same size as input
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (shortside_len, w*shortside_len//h),
                           false_fn=lambda: (shortside_len*h//w, shortside_len))
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, size=(new_h, new_w))

    xmin, ymin, xmax, ymax, label = tf.unstack(gtbox_tensor,axis=1)

    xmin, xmax = xmin*new_w//w, xmax*new_w//w
    ymin, ymax = ymin*new_h//h, ymax*new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0)
    return img_tensor, tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label]))


def flip_left_right(img_tensor, gtbox_tensor):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.image.flip_left_right(img_tensor)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtbox_tensor,axis=1)
    new_xmin, new_xmax = w-xmax, w-xmin
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax, label]))


def random_flip_left_right(img_tensor, gtbox_tensor):
    pred = tf.less(tf.constant(np.random.rand()), tf.constant(0.5))
    img_tensor, gtbox_tensor = tf.cond(pred,
                                       lambda: flip_left_right(img_tensor, gtbox_tensor),
                                       lambda: (img_tensor, gtbox_tensor)
                                       )
    return img_tensor, gtbox_tensor
