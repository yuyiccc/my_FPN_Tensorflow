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
    num_type_anchor = 9#tf.shape(rpn.anchor_scales)[0]*tf.shape(rpn.anchor_ratios)[0]
    num_anchor = tf.shape(rpn.anchors)[0]
    num_anchor_in_one_type = num_anchor//num_type_anchor

    image_with_anchor_list = []

    for i in range(num_type_anchor):
        index_start, index_end = i*num_anchor_in_one_type, (i+1)*num_anchor_in_one_type
        image_with_anchor = draw_anchor_in_image(image, rpn.anchors[index_start:index_end])
        image_with_anchor_list.append(image_with_anchor)
    tf.summary.histogram('anchors/boxes_regression', rpn.rpn_encode_boxes)
    tf.summary.histogram('anchors/boxes_scores', rpn.rpn_scores)
    return image_with_anchor_list
