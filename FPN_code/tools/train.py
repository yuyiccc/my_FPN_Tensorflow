# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:05

import tensorflow as tf
import sys
sys.path.append("../")
from data.IO.read_tfrecord import Read_tfrecord
from libs.box_utils.show_boxes import draw_box_with_tensor
import configs.global_cfg as cfg
from  tools.assist_tools import ShowProcess, check_and_create_paths
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn, debug_rpn
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
                                                  tf.reshape(gtboxes_label, [-1, 5]),
                                                  text=img_name)

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
                                gtboxes_and_label=gtboxes_label,
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
                                max_proposal_num=cfg.MAX_PROPOSAL_NUM,
                                rpn_iou_positive_threshold=cfg.RPN_IOU_POSITIVE_THRESHOLD,
                                rpn_iou_negtive_threshold=cfg.RPN_IOU_NEGATIVE_THRESHOLD,
                                rpn_mini_batchsize=cfg.RPN_MINI_BATCH_SIZE,
                                rpn_positive_ratio=cfg.POSITIVE_RATIO,
                                remove_outside_anchors=cfg.IS_FILTER_OUTSIDE_ANCHORS,
                                rpn_weight_decay=cfg.WEIGHT_DECAY
                                )
        # rpn_proposals_boxes, rpn_proposals_scores = rpn_net.rpn_proposals()


        ###########
        # summary #
        ###########
        tf.summary.image('images/gtboxes', gtboxes_in_img)

        if debug:
            # bcckbone network
            for key in end_point.keys():
                tf.summary.histogram('value/'+key, end_point[key])
            # weights
            for weight in slim.get_model_variables():
                tf.summary.histogram('weight/'+weight.name, weight.value())
            # rpn anchor
            image_with_anchor_list, anchors_list, num_anchor_list = debug_rpn.debug_rpn(rpn_net, img)
            for i, image_with_anchor in enumerate(image_with_anchor_list):
                tf.summary.image('anchors/image_with_anchors_'+str(i), image_with_anchor[0])

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
            process = ShowProcess(100)

            for i in range(100):
                process.show_process()
                if i % 100 == 0:
                    anchors_a = sess.run(rpn_net.anchors)
                    anchors_list_a = sess.run(anchors_list)
                    num_anchor_list_a = sess.run(num_anchor_list)
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                    # img_name_i, img_i, gtboxes_label_i, num_gtbox_i = sess.run((img_name, img, gtboxes_label, num_gtbox))
                    # print(img_name_i[0].decode(), i)
                    # print(img_i.shape)
                    # print(gtboxes_label_i, num_gtbox_i)
                    # print(img_i[:4, :4, 0], '\n\n')


if __name__ == '__main__':
    train()
