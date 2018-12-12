# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/6 17:13
import tensorflow as tf
from tensorflow.contrib import slim
import sys
sys.path.append('../../')
from libs.box_utils.encode_and_decode import encode_boxes,decode_boxes
from libs.box_utils import boxes_utils
from libs.box_utils.iou import calculate_iou


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
                 num_cls,
                 scale_factors,
                 fast_rcnn_nms_iou_threshold,
                 max_num_per_class,
                 fast_rcnn_score_threshold,
                 fast_rcnn_positive_threshold_iou,
                 fast_rcnn_minibatch_size,
                 fast_rcnn_positive_ratio
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
        :param num_cls: int type
        :param scale_factors [4,]
        :param fast_rcnn_nms_iou_threshold :float type
        :param max_num_per_class: int type
        :param fast_rcnn_score_threshold: float type, thresh the low score boxes
        :param fast_rcnn_positive_threshold_iou  :proposal which greater this threshold is a positive sample
        :param fast_rcnn_minibatch_size
        :param fast_rcnn_positive_ratio
        '''
        self.img_batch = img_batch
        self.img_shape = tf.shape(img_batch)
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
        self.scale_factors = scale_factors
        self.fast_rcnn_nms_iou_threshold = fast_rcnn_nms_iou_threshold
        self.max_num_per_class = max_num_per_class
        self.fast_rcnn_score_threshold = fast_rcnn_score_threshold
        self.fast_rcnn_positive_threshold_iou = fast_rcnn_positive_threshold_iou
        self.fast_rcnn_minibatch_size = fast_rcnn_minibatch_size
        self.fast_rcnn_positive_ratio = fast_rcnn_positive_ratio
        self.max_num_positive = tf.cast(self.fast_rcnn_minibatch_size * self.fast_rcnn_positive_ratio, tf.int32)

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
                img_h, img_w = tf.cast(self.img_shape[1], tf.float32), tf.cast(self.img_shape[2], tf.float32)
                ymin /= img_h
                xmin /= img_w
                ymax /= img_h
                xmax /= img_w

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

    def fast_rcnn_prediction(self):
        '''
        :param: self.fast_rcnn_cls_scores, [2000, num_cls+1], num_cls+background
        :param: self.fast_rcnn_encode_boxes, [2000, num_cls*4]
        :return: fast_rcnn_decode_boxes, [-1, 4]
        :return: fast_rcnn_category, [-1, ]
        :return: fast_rcnn_scores, [-1, ]
        :return: num_object, [-1, ]
        '''
        with tf.variable_scope('fast_rcnn_predict'):
            fast_rcnn_softmax_score = slim.softmax(self.fast_rcnn_cls_scores)

            fast_rcnn_encode_boxes = tf.reshape(self.fast_rcnn_encode_boxes, [-1, 4])
            fast_rcnn_reference_boxes = tf.tile(self.rois_boxes, [1, self.num_cls])
            fast_rcnn_reference_boxes = tf.reshape(fast_rcnn_reference_boxes, [-1, 4])

            # ues encode boxes to decode the reference boxes
            fast_rcnn_decode_boxes = decode_boxes(encode_boxes=fast_rcnn_encode_boxes,
                                                  reference_boxes=fast_rcnn_reference_boxes,
                                                  scale_factors=self.scale_factors)
            # clip decode boxes to image shape
            fast_rcnn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(boxes=fast_rcnn_decode_boxes,
                                                                              img_shape=self.img_shape)

            # mutil-class nms
            fast_rcnn_decode_boxes = tf.reshape(fast_rcnn_decode_boxes, [-1, 4*self.num_cls])
            fast_rcnn_decode_boxes, fast_rcnn_category, fast_rcnn_scores, num_object = \
                self.mutil_class_nms(boxes=fast_rcnn_decode_boxes,
                                     scores=fast_rcnn_softmax_score)
            return fast_rcnn_decode_boxes, fast_rcnn_category, fast_rcnn_scores, num_object

    def mutil_class_nms(self, boxes, scores):
        '''
        :param boxes: [N, num_cls*4]
        :param scores:  [N, num_cls+1]
        :return:
        '''
        category = tf.argmax(scores, axis=1)
        object_mask = tf.cast(tf.not_equal(category, 0), tf.float32)

        boxes = boxes * tf.expand_dims(object_mask, axis=1)
        scores = scores * tf.expand_dims(object_mask, axis=1)

        boxes = tf.reshape(boxes, [-1, self.num_cls, 4])

        boxes_list = tf.unstack(boxes, axis=1)
        scores_list = tf.unstack(scores[:, 1:], axis=1)

        filter_boxes_list = []
        filter_category_list = []
        filter_scores_list = []

        for boxes_i, scores_i in zip(boxes_list, scores_list):
            indices_i = boxes_utils.non_maximal_suppression(boxes=boxes_i,
                                                            scores=scores_i,
                                                            iou_threshold=self.fast_rcnn_nms_iou_threshold,
                                                            max_output_size=self.max_num_per_class,
                                                            name='fast_rcnn_nms')
            filter_boxes_list.append(tf.gather(boxes_i, indices_i))
            filter_category_list.append(tf.gather(category, indices_i))
            filter_scores_list.append(tf.gather(scores_i, indices_i))

        filter_boxes = tf.concat(filter_boxes_list, axis=0)
        filter_category = tf.concat(filter_category_list, axis=0)
        filter_scores = tf.concat(filter_scores_list, axis=0)

        # filter low scores boxes
        scores_indices = tf.reshape(tf.where(tf.greater(filter_scores, self.fast_rcnn_score_threshold)), [-1])
        filter_boxes = tf.gather(filter_boxes, scores_indices)
        filter_category = tf.gather(filter_category, scores_indices)
        filter_scores = tf.gather(filter_scores, scores_indices)

        num_object = tf.shape(scores_indices)

        return filter_boxes, filter_category, filter_scores, num_object

    def fast_rcnn_loss(self):
        '''
        :return:
        '''
        self.make_minibatch()
        pass

    def make_minibatch(self):
        proposal_matched_boxes, proposal_matched_label, object_mask = \
            self.match_predict_and_gtboxes()

        positive_indices = tf.reshape(tf.where(tf.equal(object_mask, 1)), [-1])
        true_num_positive = tf.shape(positive_indices)[0]
        num_positive = tf.where(tf.less(self.max_num_positive, true_num_positive),
                                self.max_num_positive,
                                true_num_positive)
        positive_indices = tf.random_shuffle(positive_indices)
        positive_indices = tf.slice(positive_indices,
                                    begin=[0],
                                    size=[num_positive])

        num_negative = tf.cast(self.fast_rcnn_minibatch_size - num_positive, tf.int32)
        negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0)), [-1])
        negative_indices = tf.random_shuffle(negative_indices)
        negative_indices = tf.slice(negative_indices,
                                    begin=[0],
                                    size=[num_negative])
        minibatch_indices = tf.cast(
            tf.random_shuffle(tf.concat([positive_indices, negative_indices], axis=0)),
            tf.int32)

        minibatch_matched_boxes = tf.gather(proposal_matched_boxes, minibatch_indices)
        minibatch_matched_label = tf.gather(proposal_matched_label, minibatch_indices)

        # check this function's return
        tf.summary.tensor_summary('minibatch_indices', minibatch_indices)
        tf.summary.tensor_summary('minibatch_matched_boxes', minibatch_matched_boxes)
        tf.summary.tensor_summary('minibatch_matched_label', minibatch_matched_label)

        return minibatch_indices, minibatch_matched_boxes, minibatch_matched_label

    def match_predict_and_gtboxes(self):
        '''
        :param
        :return:
        '''
        gt_boxes = tf.cast(self.gtboxes_and_label[:, :4], tf.float32)
        gt_label = self.gtboxes_and_label[:, -1]

        iou_proposal_gtboxes = calculate_iou(self.rpn_proposal_boxes, gt_boxes)

        #
        max_iou_per_proposal = tf.reduce_max(iou_proposal_gtboxes, axis=1)

        match_indices = tf.cast(tf.argmax(iou_proposal_gtboxes, axis=1), tf.int32)

        # [2000, 4]
        proposal_matched_boxes = tf.gather(gt_boxes, match_indices)

        positive_mask = tf.greater(max_iou_per_proposal, self.fast_rcnn_positive_threshold_iou)
        object_mask = tf.cast(positive_mask, tf.int32)

        # make negative sample's cls label to background
        proposal_matched_label = tf.gather(gt_label, match_indices)
        proposal_matched_label = proposal_matched_label * object_mask

        # check those return
        # tf.summary.tensor_summary('proposal_matched_boxes', proposal_matched_boxes)
        # tf.summary.tensor_summary('proposal_matched_label', proposal_matched_label)
        # tf.summary.tensor_summary('object_mask', object_mask)
        tf.summary.tensor_summary('max_iou_per_proposal', max_iou_per_proposal)

        return proposal_matched_boxes, proposal_matched_label, object_mask