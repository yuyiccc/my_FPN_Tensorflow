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
            gtboxes_in_img = draw_box_with_tensor(img, tf.reshape(gtboxes_label, [-1, 5]), text=img_name)

        ####################
        # backbone network #
        ####################

        _, end_point = get_network_byname(net_name='resnet_v1_101',
                                          inputs=img,
                                          num_classes=None,
                                          is_training=True,
                                          global_pool=False,
                                          output_stride=None,
                                          spatial_squeeze=False)


        ###########
        # summary #
        ###########
        tf.summary.image('images/gtboxes', gtboxes_in_img)

        if debug:
            # bcckbone network
            for key in end_point.keys():
                tf.summary.histogram(key,end_point[key])

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

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(iterator.initializer)
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
            process = ShowProcess(10000)
            # end_point_i = sess.run(end_point)
            for i in range(200):
                process.show_process()
                if i % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                    # img_name_i, img_i, gtboxes_label_i, num_gtbox_i = sess.run((img_name, img, gtboxes_label, num_gtbox))
                    # print(img_name_i[0].decode(), i)
                    # print(img_i.shape)
                    # print(gtboxes_label_i, num_gtbox_i)
                    # print(img_i[:4, :4, 0], '\n\n')


if __name__=='__main__':
    train()