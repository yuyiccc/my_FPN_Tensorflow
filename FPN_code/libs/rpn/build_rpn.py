# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/22 19:18

import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../../')
from libs.box_utils import make_anchor, boxes_utils, encode_and_decode, iou
from libs.losses import losses
from libs.box_utils.show_boxes import draw_box_with_tensor


class RPN(object):
    def __init__(self,
                 net_name,
                 inputs,
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
        :param inputs:  tensor type, image batch [1,,h,w,3]
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
        self.img_batch = inputs
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
                feature_maps_dict = {
                    'C2': self.end_point['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # stride=4
                    'C3': self.end_point['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # stride=8
                    'C4': self.end_point['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # stride=16
                    'C5': self.end_point['resnet_v1_50/block4']  # stride=32
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict = {
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
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
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
                    p, c = feature_pyramid['P%s' % (layer+1)], self.feature_map_dict['C%s' % layer]
                    shape_c = tf.shape(c)
                    upsample_p = tf.image.resize_nearest_neighbor(p,
                                                                  [shape_c[1], shape_c[2]],
                                                                  name='build_P%s/upsample_P' % layer)
                    c = slim.conv2d(c, num_outputs=256,
                                    kernel_size=[1, 1],
                                    stride=1,
                                    scope='build_P%s/reduce_C_dimension' % layer)
                    p = upsample_p + c
                    p = slim.conv2d(p, num_outputs=256,
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
                valid_indices = boxes_utils.filter_outside_boxes(anchors=anchors,
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
                    feature_h, feature_w = feature_map_shape[1], feature_map_shape[2]

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
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                for level in self.level:
                    if self.share_head:
                        reuse = None if level == 'P2' else True
                        scope_list = ['conv2d_3x3', 'rpn_classifier', 'rpn_regressor']
                    else:
                        reuse = None
                        scope_list = ['conv2d_3x3_'+level, 'rpn_classifier_'+level, 'rpn_regressor_'+level]
                    rpn_conv2d_3x3 = slim.conv2d(self.feature_pyramid[level], num_outputs=512,
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
                                             activation_fn=None,
                                             reuse=reuse)
                    rpn_encode_boxes = slim.conv2d(rpn_conv2d_3x3,
                                                   num_outputs=self.num_anchor_per_location*4,
                                                   kernel_size=[1, 1],
                                                   stride=1,
                                                   scope=scope_list[2],
                                                   activation_fn=None,
                                                   reuse=reuse)
                    rpn_scores_list.append(tf.reshape(rpn_scores, shape=[-1, 2]))
                    rpn_encode_boxes_list.append(tf.reshape(rpn_encode_boxes, shape=[-1, 4]))
                all_rpn_scores = tf.concat(rpn_scores_list, axis=0)
                all_rpn_encode_boxes = tf.concat(rpn_encode_boxes_list, axis=0)
                return all_rpn_encode_boxes, all_rpn_scores

    def rpn_proposals(self):
        '''
        :param:self.anchors: shape:[-1, 4]->[ymin, xmin, ymax, xmax]
        :param:self.rpn_scores: shape:[-1, 2]->[backgroud, foreground]
        :param:self.rpn_encode_boxes: shape:[-1, 4]->[ycenter, xcenter, h, w]
        :return: valid_boxes [2000, 4]
        :return: valid_scores [2000,]
        '''
        with tf.variable_scope('rpn_proposals'):
            rpn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=self.rpn_encode_boxes,
                                                              reference_boxes=self.anchors,
                                                              scale_factors=self.scale_factors)
            if not self.is_training:
                image_shape = tf.shape(self.img_batch)
                rpn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(rpn_decode_boxes, image_shape)

            rpn_softmax_scores = slim.softmax(self.rpn_scores)
            rpn_object_score = rpn_softmax_scores[:, 1]

            if self.top_k_nms:
                rpn_object_score, top_k_indices = tf.nn.top_k(rpn_object_score, k=self.top_k_nms)
                rpn_decode_boxes = tf.gather(rpn_decode_boxes, top_k_indices)

            nms_indices = boxes_utils.non_maximal_suppression(rpn_decode_boxes,
                                                              rpn_object_score,
                                                              self.rpn_nms_iou_threshold,
                                                              self.max_proposal_num)
            valid_scores = tf.gather(rpn_object_score, nms_indices)
            valid_boxes = tf.gather(rpn_decode_boxes, nms_indices)

        return valid_boxes, valid_scores

    def rpn_loss(self):
        '''
        :param: self.gtboxes_and_label: [n, 5]->[ymin, xmin, ymax, xmax, cls]
        :param: self.anchors: [m, 4]-> [ymin, xmin, ymax, xmax]
        :param:self.rpn_encode_boxes: [m, 4]->[ycenter, xcenter, h, w]
        :return:
        '''
        with tf.variable_scope('rpn_loss'):
            minibatch_indices,\
            minibatch_anchor_matched_gtboxes,\
            object_mask,\
            minibatch_label_onehot = self.make_minibatch()

            minibatch_anchors = tf.gather(self.anchors, minibatch_indices)
            minibatch_rpn_encode_boxes = tf.gather(self.rpn_encode_boxes, minibatch_indices)
            minibatch_rpn_scores = tf.gather(self.rpn_scores, minibatch_indices)

            minibatch_encode_boxes_label = encode_and_decode.encode_boxes(minibatch_anchors,
                                                                          minibatch_anchor_matched_gtboxes,
                                                                          self.scale_factors)
            # summary
            positive_anchors_in_img = draw_box_with_tensor(img_batch=self.img_batch,
                                                           boxes=minibatch_anchors*tf.expand_dims(object_mask, 1),
                                                           text=tf.shape(tf.where(tf.equal(object_mask, 1)))[0])
            negative_mask = tf.cast(tf.logical_not(tf.cast(object_mask, tf.bool)), tf.float32)
            negative_anchors_in_img = draw_box_with_tensor(img_batch=self.img_batch,
                                                           boxes=minibatch_anchors*tf.expand_dims(negative_mask, 1),
                                                           text=tf.shape(tf.where(tf.equal(negative_mask, 1)))[0])

            minibatch_decode_anchors = encode_and_decode.decode_boxes(encode_boxes=minibatch_rpn_encode_boxes,
                                                                      reference_boxes=minibatch_anchors,
                                                                      scale_factors=self.scale_factors)
            # clip boxes into image shape
            minibatch_decode_anchors = boxes_utils.clip_boxes_to_img_boundaries(minibatch_decode_anchors,
                                                                                tf.shape(self.img_batch))
            positive_decode_anchor_in_img = \
                draw_box_with_tensor(img_batch=self.img_batch,
                                     boxes=minibatch_decode_anchors*tf.expand_dims(object_mask, 1),
                                     text=tf.shape(tf.where(tf.equal(object_mask, 1)))[0]
                                     )

            tf.summary.image('images/rpn/losses/anchors_positive_minibatch', positive_anchors_in_img)
            tf.summary.image('images/rpn/losses/anchors_negative_minibatch', negative_anchors_in_img)
            tf.summary.image('images/rpn/losses/decode_anchor_positive', positive_decode_anchor_in_img)

            # losses
            with tf.variable_scope('rpn_localization_losses'):
                classify_loss = slim.losses.softmax_cross_entropy(logits=minibatch_rpn_scores,
                                                                  onehot_labels=minibatch_label_onehot)

                location_loss = losses.l1_smooth_losses(predict_boxes=minibatch_rpn_encode_boxes,
                                                        gtboxes=minibatch_encode_boxes_label,
                                                        object_weights=object_mask)
                slim.losses.add_loss(location_loss)  # add location loss to losses collections

            return location_loss, classify_loss

    def make_minibatch(self):
        with tf.variable_scope('rpn_minibatch'):
            label, anchor_matched_gtboxes, object_mask = self.rpn_find_positive_negtive_samples()
            positive_indices = tf.reshape(tf.where(tf.equal(label, 1.0)), [-1])
            num_of_positive = tf.minimum(tf.shape(positive_indices)[0],
                                         tf.cast(self.rpn_positive_ratio*self.rpn_mini_batchsize, dtype=tf.int32)
                                         )
            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices,
                                        begin=[0],
                                        size=[num_of_positive])
            negative_indices = tf.reshape(tf.where(tf.equal(label, 0.0)), [-1])
            num_of_negative = tf.minimum(tf.shape(negative_indices)[0],
                                         self.rpn_mini_batchsize-num_of_positive)
            negative_indices = tf.random_shuffle(negative_indices)
            negative_indices = tf.slice(negative_indices,
                                        begin=[0],
                                        size=[num_of_negative])
            minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_anchor_matched_gtboxes = tf.gather(anchor_matched_gtboxes, minibatch_indices)
            object_mask = tf.gather(object_mask, minibatch_indices)
            label = tf.cast(tf.gather(label, minibatch_indices), dtype=tf.int32)
            onehot_label = tf.one_hot(label, depth=2)

            return minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, onehot_label

    def rpn_find_positive_negtive_samples(self):
        with tf.variable_scope('find_positive_negtive_samples'):
            gtboxes = tf.reshape(self.gtboxes_and_label[:, :-1], [-1, 4])
            gtboxes = tf.cast(gtboxes, dtype=tf.float32)

            anchors_gtboxes_iou = iou.calculate_iou(self.anchors, gtboxes)  # [m,n]

            max_iou_each_anchors = tf.reduce_max(anchors_gtboxes_iou, axis=1)

            # -1 means ignore 0 means negative 1 means positive
            labels = tf.ones(shape=[tf.shape(self.anchors)[0], ], dtype=tf.float32)*(-1)

            matchs = tf.cast(tf.argmax(anchors_gtboxes_iou, axis=1), tf.int32)
            anchors_matched_gtboxes = tf.gather(gtboxes, matchs)

            # an anchor that has IOU higher than 0.7 with any gtboxes
            positives_1 = tf.greater_equal(max_iou_each_anchors, self.rpn_iou_positive_threshold)

            # avoid there is gtbox that no anchors matched
            max_iou_each_gtboxes = tf.reduce_max(anchors_gtboxes_iou, axis=0)

            positives_2 = tf.reduce_sum(tf.cast(tf.equal(anchors_gtboxes_iou, max_iou_each_gtboxes), tf.float32), axis=1)

            positives = tf.logical_or(positives_1, tf.cast(positives_2, tf.bool))

            negative = tf.less(max_iou_each_anchors, self.rpn_iou_negtive_threshold)
            negative = tf.logical_and(negative, tf.greater_equal(max_iou_each_anchors, 0.1))

            labels += 2*tf.cast(positives, dtype=tf.float32)  # now, positive is 1, ignore is -1
            labels += tf.cast(negative, dtype=tf.float32)  # positive 1.0or2.0, negative is 0.0, ignore is -1.0

            # make positive 2.0  equal to 1.0
            positives = tf.cast(tf.greater_equal(labels, 1.0), dtype=tf.float32)
            ignore = tf.cast(tf.equal(labels, -1.0), dtype=tf.float32)*(-1)
            labels = positives + ignore

            object_mask = tf.cast(positives, dtype=tf.float32)

            return labels, anchors_matched_gtboxes, object_mask




