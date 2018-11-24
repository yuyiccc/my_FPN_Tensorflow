# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/22 19:18

import tensorflow as tf
import tensorflow.contrib.slim as slim



class RPN(object):
    def __init__(self,
                 net_name,
                 input,
                 gtboxes_and_label,
                 is_training,
                 end_point,
                 anchor_scales,
                 anchor_ratios,
                 scale_factors,
                 base_anchor_size_list,
                 stride,
                 level,
                 top_k_nms,
                 share_head=False,
                 rpn_nms_iou_threshold=0.7,
                 max_proposal_num=300,
                 rpn_iou_positive_threshold=0.7,
                 rpn_iou_negtive_threshold=0.3,
                 rpn_mini_batchsize=256,
                 rpn_positive_ratio=0.5,
                 remove_outside_anchors=False,
                 rpn_weight_decay=1e-4
                 ):
        '''
        :param net_name: string type, backbone network name , 'resnet_v1_50'
        :param input:  tensor type, image batch [1,,h,w,3]
        :param gtboxes_and_label: tensor type, [-1,5],[ymin, xmin, ymax, xmax, label]
        :param is_training:  bool type, training or test
        :param end_point:   dict type, backbone network's feature maps
        :param anchor_scales:  list type, anchor's scales
        :param anchor_ratios:  list type, anchor's ratio
        :param scale_factors:  list type, scale factors
        :param base_anchor_size_list: list type, based anchor size
        :param stride:  list type, feature maps' stride
        :param level: list type, use what feature map
        :param top_k_nms:  int type, top k boxes whill keep
        :param share_head: bool type, if share the head or not
        :param rpn_nms_iou_threshold: float type, rpn nms' iou threshold
        :param max_proposal_num: int type, max number of proposal box
        :param rpn_iou_positive_threshold: float type, iou threshold for positive proposal
        :param rpn_iou_negtive_threshold:float type, iou threshold for negtive proposal
        :param rpn_mini_batchsize: int type,  mini batch size
        :param rpn_positive_ratio: float type, positive proposal ratio in mini batch
        :param remove_outside_anchors: bool type, remove the outside anchors or not
        :param rpn_weight_decay: rpn network's weight decay
        '''
        self.net_name = net_name
        self.img_batch = input
        self.gtboxes_and_label = gtboxes_and_label
        self.is_training = is_training
        self.end_point = end_point
        self.anchor_scales = tf.constant(anchor_scales, dtype=tf.float32)
        self.anchor_ratio = tf.constant(anchor_ratios, dtype=tf.float32)
        self.scale_factors = scale_factors
        self.share_head = share_head
        self.num_anchor_per_location = len(anchor_scales)*len(anchor_ratios)
        self.base_anchor_size_list = base_anchor_size_list
        self.stride = stride
        self.level = level
        self.top_k_nms = top_k_nms
        self.rpn_nms_iou_threshold = rpn_nms_iou_threshold
        self.max_proposal_num = max_proposal_num
        self.rpn_iou_positive_threshold = rpn_iou_positive_threshold
        self.rpn_iou_negtive_threshold = rpn_iou_negtive_threshold
        self.rpn_mini_batchsize = rpn_mini_batchsize
        self.rpn_positive_ratio = rpn_positive_ratio
        self.remove_outside_anchors = remove_outside_anchors
        self.rpn_weight_decay = rpn_weight_decay

        self.feature_map_dict = self.get_feature_maps()
        self.feature_pyramid = self.build_feature_pyramid()

    def get_feature_maps(self):
        with tf.variable_scope('get_feature_maps'):
            if self.net_name == 'resnet_v1_50':
                feature_maps_dict={
                    'C2': self.end_point['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # stride=4
                    'C3': self.end_point['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # stride=8
                    'C4': self.end_point['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # stride=16
                    'C5': self.end_point['resnet_v1_50/block4']  # stride=32
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict={
                    'C2': self.end_point['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # stride=4
                    'C3': self.end_point['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # stride=8
                    'C4': self.end_point['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # stride=16
                    'C5': self.end_point['resnet_v1_101/block4']  # stride=32
                }
            else:
                raise Exception('get no feature maps')
            return feature_maps_dict

    def build_feature_pyramid(self):

        feature_pyramid = {}
        with tf.variable_scope('build_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=self.rpn_weight_decay):
                feature_pyramid[]





        return feature_pyramid