# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/22 19:18

import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../../')
from libs.box_utils import show_boxes, make_anchor, boxes_utils


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
        self.anchor_ratios = tf.constant(anchor_ratios, dtype=tf.float32)
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
        self.anchors, self.rpn_encode_boxes, self.rpn_scores = self.get_anchors_and_rpn_predict()

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
                feature_pyramid['P5'] = slim.conv2d(self.feature_map_dict['C5'],
                                                    num_outputs=256,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='build_P5')
                # P6 is downsample of P5
                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                        kernel_size=[2, 2],
                                                        stride=2,
                                                        scope='build_P6')
                for layer in range(4, 1, -1):
                    p, c = feature_pyramid['P%s' % (layer+1)], self.feature_map_dict['C%' % layer]
                    shape_c = tf.shape(c)
                    upsample_p = tf.image.resize_nearest_neighbor(p,
                                                                  [shape_c[1], shape_c[2]],
                                                                  name='build_P%s/upsample_P' % layer)
                    c = slim.conv2d(c, num_outputs=256,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    scope='build_P%s/reduce_C_dimension' % layer)
                    p = upsample_p + c
                    p = slim.conv2d(p, num_ouputs=256,
                                    kernel_size=[3, 3],
                                    stride=1,
                                    padding='SAME',
                                    scope='build_P%s/conv_after_plus' % layer)
                    feature_pyramid['P%s' % layer] = p
        return feature_pyramid

    def get_anchors_and_rpn_predict(self):
        anchors = self.make_anchors()
        rpn_encode_boxes, rpn_scores = self.rpn_net()

        with tf.name_scope('get_anchors_and_rpn_predict'):
            if self.remove_outside_anchors:
                valid_indices = boxes_utils.filter_outside_boxes(boxes=anchors,
                                                                 img_h=tf.shape(self.img_batch)[1],
                                                                 img_w=tf.shape(self.img_batch)[2])
                valid_anchors = tf.gather(anchors, valid_indices)
                valid_rpn_encode_boxes = tf.gather(rpn_encode_boxes, valid_indices)
                valid_rpn_scores = tf.gather(rpn_scores, valid_indices)

                return valid_anchors, valid_rpn_encode_boxes, valid_rpn_scores
            else:
                return anchors, rpn_encode_boxes, rpn_scores

    def make_anchors(self):
        '''
        :return: [-1,4] ,all level's anchors
        '''
        with tf.variable_scope('make_anchors'):
            anchor_list = []

            with tf.name_scope('make_anchors_all_level'):
                for level, base_anchor_size, stride in zip(self.level, self.base_anchor_size_list, self.stride):

                    feature_map_shape = tf.shape(self.feature_pyramid[level])
                    feature_h, feature_w = feature_map_shape[0], feature_map_shape[1]

                    temp_anchors = make_anchor.make_anchors(base_anchor_size, self.anchor_scales,
                                                            self.anchor_ratios,
                                                            feature_h,
                                                            feature_w,
                                                            stride,
                                                            name='make_anchor_{}'.format(level))
                    temp_anchors = tf.reshape(temp_anchors, [-1, 4])
                    anchor_list.append(temp_anchors)

                all_level_anchors = tf.concat(anchor_list, axis=0)
                return all_level_anchors

    def rpn_net(self):
        '''
        :return: rpn_encode_boxes:[-1,4]  , boxes regression value
        :return: rpn_scores: [-1,2], boxes fore-grand and back-grand scores
        '''
        rpn_encode_boxes_list = []
        rpn_scores_list = []
        with tf.variable_scope('rpn_net'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2regularizer(self.rpn_weight_decay)):
                for level in self.level:
                    if self.share_head:
                        reuse = None if level == 'P2' else True
                        scope_list = ['conv2d_3x3', 'rpn_classifier', 'rpn_regressor']
                    else:
                        reuse = None
                        scope_list = ['conv2d_3x3_'+level, 'rpn_classifier_'+level, 'rpn_regressor_'+level]
                    rpn_conv2d_3x3 = slim.conv2d(self.feature_pyramid[level],
                                                 num_ouputs=512,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[0],
                                                 reuse=reuse
                                                 )
                    rpn_scores = slim.conv2d(rpn_conv2d_3x3,
                                             num_outputs=self.num_anchor_per_location*2,
                                             kernel_size=[1, 1],
                                             stride=1,
                                             scope=scope_list[1],
                                             reuse=reuse)
                    rpn_encode_boxes = slim.conv2d(rpn_conv2s_3x3,
                                                   num_outputs=self.num_anchor_per_location*4,
                                                   kernel_size=[1, 1],
                                                   stride=1,
                                                   scope=scope_list[2],
                                                   reuse=reuse)
                    rpn_scores_list.append(tf.reshape(rpn_scores, shape=[-1, 2]))
                    rpn_encode_boxes_list.append(tf.reshape(rpn_encode_boxes, shape=[-1, 4]))
                all_rpn_scores = tf.concat(rpn_scores_list, axis=0)
                all_rpn_encode_boxes = tf.concat(rpn_encode_boxes_list, axis=0)
                return all_rpn_encode_boxes, all_rpn_scores

