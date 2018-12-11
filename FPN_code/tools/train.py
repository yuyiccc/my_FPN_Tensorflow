# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:05

import tensorflow as tf
import sys
sys.path.append("../")
from data.IO.read_tfrecord import Read_tfrecord
from libs.box_utils.show_boxes import draw_box_with_tensor, draw_boxes_with_category
import configs.global_cfg as cfg
from  tools.assist_tools import ShowProcess, check_and_create_paths
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn, debug_rpn
from libs.fast_rcnn import build_fast_rcnn
from  tools.restore_model import get_restorer
import tensorflow.contrib.slim as slim




debug = True


def train():
    with tf.Graph().as_default():

        ##############
        # input data #
        ##############

        with tf.name_scope('get_batch'):
            data = Read_tfrecord()
            iterator, img_name, img, gtboxes_label, num_gtbox = data.get_batch_data()

        with tf.name_scope('draw_gtboxes'):
            gtboxes_in_img = draw_box_with_tensor(img,
                                                  tf.reshape(gtboxes_label, [-1, 5])[:, :-1],
                                                  text=img_name)
        # original_img = tf.squeeze(img, axis=0)+tf.constant(cfg.DEPTH_MEAN)
        # original_img = tf.reshape(original_img, shape=tf.shape(img))
        # tf.summary.image('images/original_images', original_img)


        ####################
        # backbone network #
        ####################

        _, end_point = get_network_byname(net_name=cfg.NETWORK_NAME,
                                          inputs=img,
                                          num_classes=None,
                                          is_training=True,
                                          global_pool=False,
                                          output_stride=None,
                                          spatial_squeeze=False)

        ###############
        # rpn network #
        ###############
        rpn_net = build_rpn.RPN(net_name=cfg.NETWORK_NAME,
                                inputs=img,
                                gtboxes_and_label=tf.squeeze(gtboxes_label, axis=0),
                                is_training=True,
                                end_point=end_point,
                                anchor_scales=cfg.ANCHOR_SCALES,
                                anchor_ratios=cfg.ANCHOR_RATIOS,
                                scale_factors=cfg.SCALE_FACTOR,
                                base_anchor_size_list=cfg.BASE_ANCHOR_SIZE_LIST,
                                stride=cfg.STRIDE,
                                level=cfg.LEVEL,
                                top_k_nms=cfg.TOP_K_NMS,
                                share_head=cfg.IS_SHARE_HEAD,
                                rpn_nms_iou_threshold=cfg.RPN_NMS_IOU_THRESHOLD,
                                max_proposal_num=cfg.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfg.RPN_IOU_POSITIVE_THRESHOLD,
                                rpn_iou_negtive_threshold=cfg.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batchsize=cfg.RPN_MINI_BATCH_SIZE,
                                rpn_positive_ratio=cfg.POSITIVE_RATIO,
                                remove_outside_anchors=cfg.IS_FILTER_OUTSIDE_ANCHORS,
                                rpn_weight_decay=cfg.RPN_WEIGHT_DECAY
                                )
        rpn_proposals_boxes, rpn_proposals_scores = rpn_net.rpn_proposals()
        rpn_location_loss, rpn_classification_loss = rpn_net.rpn_loss()
        rpn_net_loss = rpn_location_loss+rpn_classification_loss

        with tf.name_scope('draw_proposals'):
            rpn_object_indices = tf.reshape(tf.where(tf.greater(rpn_proposals_scores, 0.5)), shape=[-1])
            rpn_object_boxes = tf.gather(rpn_proposals_boxes,
                                         indices=rpn_object_indices)
            rpn_object_boxes_in_img = draw_box_with_tensor(img_batch=img,
                                                           boxes=rpn_object_boxes,
                                                           text='rpn_object_boxes')

            rpn_proposals_boxes_in_img = draw_box_with_tensor(img_batch=img,
                                                              boxes=rpn_proposals_boxes,
                                                              text=tf.shape(rpn_proposals_boxes)[0])

        #############
        # fast-rcnn #
        #############
        fast_rcnn = build_fast_rcnn.FastRcnn(img_batch=img,
                                             feature_dict=rpn_net.feature_pyramid,
                                             rpn_proposal_boxes=rpn_proposals_boxes,
                                             rpn_proposal_scores=rpn_proposals_scores,
                                             gtboxes_and_label=tf.squeeze(gtboxes_label, axis=0),
                                             crop_size=cfg.CROP_SIZE,
                                             roi_pooling_kernel_size=cfg.ROI_POOLING_KERNEL_SIZE,
                                             levels=cfg.LEVEL,
                                             is_training=True,
                                             weights_regularizer=cfg.FAST_RCNN_WEIGHTS_DECAY,
                                             num_cls=cfg.NUM_CLASSES,
                                             scale_factors=cfg.SCALE_FACTOR,
                                             fast_rcnn_nms_iou_threshold=cfg.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             max_num_per_class=cfg.MAX_NUM_PER_CLASS,
                                             fast_rcnn_score_threshold=cfg.FAST_RCNN_SCORE_THRESHOLD,
                                             fast_rcnn_positive_threshold_iou=cfg.FAST_RCNN_POSITIVE_THRESHOLD_IOU
                                             )
        fast_rcnn_decode_boxes, fast_rcnn_category, fast_rcnn_scores, num_object = \
            fast_rcnn.fast_rcnn_prediction()
        fast_rcnn.fast_rcnn_loss()
        fast_rcnn_prediction_in_image = draw_boxes_with_category(img_batch=img,
                                                                 boxes=fast_rcnn_decode_boxes,
                                                                 category=fast_rcnn_category,
                                                                 scores=fast_rcnn_scores)

        ###########
        # summary #
        ###########
        # ground truth boxes
        tf.summary.image('images/gtboxes', gtboxes_in_img)

        # rpn net's proposals
        tf.summary.image('images/rpn/proposals', rpn_proposals_boxes_in_img)
        tf.summary.image('images/rpn/objects', rpn_object_boxes_in_img)

        # rpn loss scale
        tf.summary.scalar('losses/rpn/location_loss', rpn_location_loss)
        tf.summary.scalar('losses/rpn/classify_loss', rpn_classification_loss)

        # fast rcnn prediction boxes
        tf.summary.image('images/fast_rcnn/prediction_boxes', fast_rcnn_prediction_in_image)

        if debug:
            # bcckbone network
            for key in end_point.keys():
                tf.summary.histogram('value/'+key, end_point[key])
            # weights
            for weight in slim.get_model_variables():
                tf.summary.histogram('weight/'+weight.name, weight.value())
            # rpn anchor
            image_with_anchor_list = debug_rpn.debug_rpn(rpn_net, img)
            for i, image_with_anchor in enumerate(image_with_anchor_list):
                tf.summary.image('anchors/image_with_anchors_'+str(i), image_with_anchor[0])
            # fast rcnn prediction
            tf.summary.tensor_summary('image_shape', tf.shape(img))
            tf.summary.tensor_summary('fast_rcnn_prediction_boxes', fast_rcnn_decode_boxes)

        summary_op = tf.summary.merge_all()
        summary_path = cfg.SUMMARY_PATH
        check_and_create_paths([summary_path])

        ################
        # session part #
        ################

        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer()
        )

        checkpoint_path, restorer = get_restorer()

        with tf.Session() as sess:

            # initial part
            sess.run(init_op)
            sess.run(iterator.initializer)
            if checkpoint_path:
                restorer.restore(sess, checkpoint_path)
                print('restore is done!!!')
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, 1)
            summary_writer.flush()


def initial_part(iterator):
    init_op = tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()
    )
    sess = tf.InteractiveSession()
    sess.run(init_op)
    sess.run(iterator.initializer)
    return sess


if __name__ == '__main__':
    train()
