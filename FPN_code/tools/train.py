# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:05

import tensorflow as tf
import sys
sys.path.append("../")
from data.IO.read_tfrecord import Read_tfrecord
from libs.box_utils.show_boxes import draw_box_with_tensor

def train():
    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            data = Read_tfrecord()
            iterator, img_name, img, gtboxes_label, num_gtbox = data.get_batch_data()

        with tf.name_scope('draw_gtboxes'):
            gtboxes_in_img = draw_box_with_tensor(img, gtboxes_label[:, :4], text=gtboxes_label[:, -1])







