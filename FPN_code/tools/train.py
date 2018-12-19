# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:05

import tensorflow as tf
import sys
import os
sys.path.append("../")
from data.IO.read_tfrecord import Read_tfrecord
from libs.box_utils.boxes_utils import  clip_boxes_to_img_boundaries
from libs.box_utils.show_boxes import draw_box_with_tensor, draw_boxes_with_category
import configs.global_cfg as cfg
from  tools.assist_tools import ShowProcess, check_and_create_paths
from libs.networks.network_factory import get_network_byname
from libs.rpn import build_rpn, debug_rpn
from libs.fast_rcnn import build_fast_rcnn
from  tools.restore_model import get_restorer
import tensorflow.contrib.slim as slim
import time



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
                                                           text=tf.shape(rpn_object_boxes)[0])
            clip_rpn_proposals_boxes = clip_boxes_to_img_boundaries(rpn_proposals_boxes,
                                                                    tf.shape(img))
            rpn_proposals_boxes_in_img = draw_box_with_tensor(img_batch=img,
                                                              boxes=clip_rpn_proposals_boxes,
                                                              text=tf.shape(rpn_proposals_boxes)[0]
                                                              )

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
                                             fast_rcnn_positive_threshold_iou=cfg.FAST_RCNN_POSITIVE_THRESHOLD_IOU,
                                             fast_rcnn_minibatch_size=cfg.FAST_RCNN_MINIBATCH_SIZE,
                                             fast_rcnn_positive_ratio=cfg.FAST_RCNN_POSITIVE_RATIO
                                             )
        fast_rcnn_decode_boxes, fast_rcnn_category, fast_rcnn_scores, num_object = \
            fast_rcnn.fast_rcnn_prediction()
        fast_rcnn_boxes_loss, fast_rcnn_cls_loss = fast_rcnn.fast_rcnn_loss()
        fast_rcnn_total_loss = fast_rcnn_boxes_loss + fast_rcnn_cls_loss

        with tf.name_scope('fast_rcnn_prediction_boxes'):
            fast_rcnn_prediction_in_image = draw_boxes_with_category(img_batch=img,
                                                                     boxes=fast_rcnn_decode_boxes,
                                                                     category=fast_rcnn_category,
                                                                     scores=fast_rcnn_scores)

        #####################
        # optimization part #
        #####################
        # global_step = tf.train.get_or_create_global_step()
        # total_loss = slim.losses.get_losses()
        # total_loss = tf.reduce_sum(total_loss * tf.constant(cfg.LOSS_WEIGHT, dtype=tf.float32))
        #
        # lr = tf.train.piecewise_constant(global_step,
        #                                  [60000],
        #                                  [cfg.BASE_LEARNING_RATE, cfg.BASE_LEARNING_RATE/10])
        #
        # optimizer = slim.train.MomentumOptimizer(learning_rate=lr,
        #                                          momentum=cfg.MOMENTUM,)
        #
        # train_op = optimizer.minimize(total_loss, global_step)

        global_step = tf.train.get_or_create_global_step()
        total_loss = slim.losses.get_total_loss()

        lr = tf.train.piecewise_constant(global_step,
                                         [60000],
                                         [cfg.BASE_LEARNING_RATE, cfg.BASE_LEARNING_RATE/10])

        optimizer = tf.train.MomentumOptimizer(learning_rate=lr,
                                               momentum=cfg.MOMENTUM)

        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)
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
        tf.summary.scalar('losses/rpn/cls_loss', rpn_classification_loss)
        tf.summary.scalar('losses/rpn/total_loss', rpn_net_loss)

        # fast rcnn prediction boxes
        tf.summary.image('images/fast_rcnn/prediction_boxes', fast_rcnn_prediction_in_image)

        # fast loss part
        tf.summary.scalar('losses/fast_rcnn/location_loss', fast_rcnn_boxes_loss)
        tf.summary.scalar('losses/fast_rcnn/cls_loss', fast_rcnn_cls_loss)
        tf.summary.scalar('losses/fast_rcnn/total_loss', fast_rcnn_total_loss)
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('learing_rate', lr)

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
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
            saver = tf.train.Saver()
            if checkpoint_path:
                restorer.restore(sess, checkpoint_path)
                print('restore is done!!!')
            step = 0
            while True:
                try:
                    if step >= 1:
                        break
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    start_time = time.time()
                    _global_step,\
                        _img_name,\
                        _rpn_location_loss,\
                        _rpn_classification_loss,\
                        _rpn_net_loss,\
                        _fast_rcnn_boxes_loss,\
                        _fast_rcnn_cls_loss,\
                        _fast_rcnn_total_loss,\
                        _total_loss,\
                        _train_op,\
                        summary_str\
                        = sess.run([global_step,
                                    img_name,
                                    rpn_location_loss,
                                    rpn_classification_loss,
                                    rpn_net_loss,
                                    fast_rcnn_boxes_loss,
                                    fast_rcnn_cls_loss,
                                    fast_rcnn_total_loss,
                                    total_loss,
                                    train_op,
                                    summary_op])
                    end_time = time.time()

                    # print the result in screen
                    if 1:  # step % 10 == 0:
                        cost_time = end_time - start_time
                        print("""-----time:%s---step:%d---image name:%s---cost_time:%.4fs-----\n
                        total_loss:%.4f\n
                        rpn_boxes_loss:%.4f         rpn_class_loss:%.4f         rpn_total_loss:%.4f\n
                        fast_rcnn_boxes_loss:%.4f   fast_rcnn_class_loss:%.4f   fast_rcnn_total_loss:%4f"""
                              % (training_time,
                                 _global_step,
                                 str(_img_name),
                                 cost_time,
                                 _total_loss,
                                 _rpn_location_loss,
                                 _rpn_classification_loss,
                                 _rpn_net_loss,
                                 _fast_rcnn_boxes_loss,
                                 _fast_rcnn_cls_loss,
                                 _fast_rcnn_total_loss)
                              )
                    # add summary
                    if 1:  # step % 100 == 0:
                        # summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                    # save ckpt
                    if step % 10000 == 0 and step > 1:
                        check_and_create_paths([cfg.CKPT_PATH])
                        save_path = os.path.join(cfg.CKPT_PATH, 'model_weights')
                        saver.save(sess, save_path, global_step)
                    step += 1
                    print(step)
                except tf.errors.OutOfRangeError:
                    break
            summary_writer.close()

# this function is for debug the function which using in trainning function
# def initial_part(iterator):
#     init_op = tf.group(
#         tf.local_variables_initializer(),
#         tf.global_variables_initializer()
#     )
#     sess = tf.InteractiveSession()
#     sess.run(init_op)
#     sess.run(iterator.initializer)
#     return sess


if __name__ == '__main__':
    train()
