# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/27 15:40

import tensorflow as tf
import sys
sys.path.append('../../')
from libs.box_utils.show_boxes import draw_anchor_in_image


def debug_rpn(rpn, image):
    '''
    :param rpn: rpn class
    :param image: image
    :return:  images with anchors
    1. images with anchors
    2. histogram of rpn boxes
    3. histogram of rpn scores
    '''
    num_level_anchor = len(rpn.level)

    image_with_anchor_list = []

    index_start = 0
    for i in range(num_level_anchor):
        feature_shape = tf.shape(rpn.feature_pyramid['P%d' % (i+2)])
        num_anchor_i = feature_shape[1]*feature_shape[2]
        if i == 0:
            index_end = num_anchor_i
        else:
            index_start, index_end = index_end, index_end+num_anchor_i
        image_with_anchor = draw_anchor_in_image(image, rpn.anchors[index_start:index_end])
        image_with_anchor_list.append(image_with_anchor)
    tf.summary.histogram('anchors/boxes_regression', rpn.rpn_encode_boxes)
    tf.summary.histogram('anchors/boxes_scores', rpn.rpn_scores)
    return image_with_anchor_list
