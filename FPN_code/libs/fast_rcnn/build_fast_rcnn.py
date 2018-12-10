# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/6 17:13
import tensorflow as tf
from tensorflow.contrib import slim

class FastRcnn(object):
    def __init__(self,
                 img_batch,
                 feature_dict,
                 rpn_proposal_boxes,
                 rpn_proposal_scores,
                 gtboxes_and_label,
                 crop_size,
                 roi_pooling_kernel_size,
                 levels,
                 is_training,
                 weights_regularizer,
                 num_cls
                 ):
        '''
        :param img_batch: [1,h,w,3]
        :param feature_dict: dictionary type [P2, P3, P4, P5, P6]
        :param rpn_proposal_boxes: [2000,4]
        :param rpn_proposal_scores:  [2000,]
        :param gtboxes_and_label:[-1,5]
        :param crop_size :int type
        :param roi_pooling_kernel_size: int
        :param levels: ['p2', 'p3',...]
        :param is_training: bool type
        :param weights_regularizer: int type
        '''
        self.img_batch = img_batch
        self.img_shape = tf.cast(tf.shape(img_batch)[1:3], dtype=tf.float32)
        self.feature_dict = feature_dict
        self.rpn_proposal_boxes = rpn_proposal_boxes
        self.rpn_proposal_scores = rpn_proposal_scores
        self.gtboxes_and_label = gtboxes_and_label
        self.crop_size = crop_size
        self.roi_pooling_kernel_size = roi_pooling_kernel_size
        self.levels = levels
        self.min_level = int(levels[0][-1])
        self.max_level = min(int(levels[-1][-1]), 5)
        self.is_training = is_training
        self.weights_regularizer = weights_regularizer
        self.num_cls = num_cls

        self.rois_feature, self.rois_boxes = self.get_rois()
        self.fast_rcnn_cls_scores, self.fast_rcnn_encode_boxes = self.fast_rcnn_net()

    def get_rois(self):
        '''
        :param: self.feature_dict
        :param: self.rpn_proposal_boxes
        :return:
        '''

        def filter_zeros_boxes(boxes):
            none_zeros_indices = tf.reshape(
                tf.where(
                    tf.not_equal(
                        tf.reduce_sum(
                            tf.cast(tf.equal(boxes, 0), tf.int32),
                            axis=1),
                        4)),
                [-1])
            none_zeros_boxes = tf.gather(boxes, none_zeros_indices)
            return none_zeros_boxes, none_zeros_indices
        levels = self.assign_levels()

        rois_feature = []
        rois_boxes = []

        with tf.variable_scope('fast_rcnn_roi'):
            for level in range(self.min_level, self.max_level+1):
                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level)), shape=[-1])
                level_i_boxes = tf.gather(self.rpn_proposal_boxes, level_i_indices)
                level_i_feature = self.feature_dict['P%d' % level]

                # in case there is none this level's proposal
                # we pack zero. After all level is done, we can filter the zero boxes and features
                level_i_boxes = tf.cond(tf.equal(tf.shape(level_i_indices)[0], 0),
                                        lambda: tf.constant([[0, 0, 0, 0]], dtype=tf.float32),
                                        lambda: level_i_boxes)
                rois_boxes.append(level_i_boxes)

                # normalize the coordinate
                ymin, xmin, ymax, xmax = tf.unstack(level_i_boxes, axis=1)
                ymin /= self.img_shape[0]
                xmin /= self.img_shape[1]
                ymax /= self.img_shape[0]
                xmax /= self.img_shape[1]

                level_i_cropped_rois = tf.image.crop_and_resize(level_i_feature,
                                                                boxes=tf.transpose(tf.stack([ymin, xmin, ymax, xmax])),
                                                                box_ind=tf.zeros_like(level_i_indices, dtype=tf.int32),
                                                                crop_size=[self.crop_size]*2)

                level_i_rois_feature = slim.max_pool2d(level_i_cropped_rois,
                                                       kernel_size=[self.roi_pooling_kernel_size]*2,
                                                       stride=self.roi_pooling_kernel_size)

                rois_feature.append(level_i_rois_feature)
            rois_feature = tf.concat(rois_feature, axis=0)
            rois_boxes = tf.concat(rois_boxes, axis=0)

            none_zero_boxes, none_zero_indices = filter_zeros_boxes(rois_boxes)
            none_zero_features = tf.gather(rois_feature, none_zero_indices)

            # check the boxes and feature
            # tf.summary.tensor_summary('none_zero_features', none_zero_features)
            # tf.summary.tensor_summary('none_zero_boxes', none_zero_boxes)
            # tf.summary.tensor_summary('rois_level', levels)

        return none_zero_features, none_zero_boxes

    def assign_levels(self):
        '''
        :param: self.rpn_rpoposal_boxes [2000, 4]
        :return: levels: [2000,]

        ues boxes size to assign it to feature pyramid
        '''
        ymin, xmin, ymax, xmax = tf.unstack(self.rpn_proposal_boxes, axis=1)
        # w,h should greater than 0.
        w, h = xmax-xmin, ymax-ymin

        levels = tf.floor(4. + tf.log(tf.sqrt(w * h)/224.)/tf.log(2.))

        levels = tf.maximum(levels, self.min_level)
        levels = tf.minimum(levels, self.max_level)

        return levels

    def fast_rcnn_net(self):
        '''
        :param: self.rois_features
        :return: fast_rcnn_cls_scores, fast_rcnn_encode_boxes

        net struct:
        two hidden 1024-d fully-connected (fc) layers
        (each followed by ReLU) before the final classification and
        bounding box regression layers
        '''
        with tf.variable_scope('fast_rcnn_net'):
            with slim.arg_scope([slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.weights_regularizer)
                                ):
                flatten_roi_features = slim.flatten(self.rois_feature)

                # two hidden 1024-d fully-connected layers
                roi_features = slim.fully_connected(inputs=flatten_roi_features,
                                                    num_outputs=1024,
                                                    scope='fc_1')
                roi_features = slim.fully_connected(inputs=roi_features,
                                                    num_outputs=1024,
                                                    scope='fc_2')
                # cls scores
                fast_rcnn_cls_score = slim.fully_connected(inputs=roi_features,
                                                           num_outputs=self.num_cls+1,
                                                           activation_fn=None,
                                                           scope='fast_rcnn_cls_scores')
                # encode boxes
                fast_rcnn_encode_boxes = slim.fully_connected(inputs=roi_features,
                                                              num_outputs=self.num_cls*4,
                                                              activation_fn=None,
                                                              scope='fast_rcnn_encode_boxes')

                # check the cls scores and encode boxes
                # tf.summary.tensor_summary('summary_fast_rcnn_encode_boxes', fast_rcnn_encode_boxes)
                # tf.summary.tensor_summary('summary_fast_rcnn_cls_scores', fast_rcnn_cls_score)

        return fast_rcnn_cls_score, fast_rcnn_encode_boxes





