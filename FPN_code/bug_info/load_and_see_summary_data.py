# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/6 15:17

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append('../')
import configs.global_cfg as cfg

summary_path = cfg.SUMMARY_PATH
summary_name = cfg.SUMMARY_PATH.split('\\')[-1]


def load_and_see_tensor(summary_path, tag_name):
    for e in tf.train.summary_iterator(summary_path):
        for v in e.summary.value:
            if v.tag in tag_name:
                fb = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                shape = []
                for d in v.tensor.tensor_shape.dim:
                    shape.append(d.size)
                fb = fb.reshape(shape)
                print(v.tag)
                print(fb)


def find_file(path):
    file_name = os.listdir(path)
    file_path = os.path.join(path, file_name[0])
    return file_path


if __name__ == '__main__':
    if summary_name == 'resnet_v1_50_pascal_debug_rpn_loss_function':
        summary_path = find_file(summary_path)
        tag_name = ['rpn_loss/rpn_localization_losses/tensor/location_l1_loss']
        load_and_see_tensor(summary_path, tag_name)
    elif summary_name == 'resnet_v1_50_pascal_debug_get_rois':
        summary_path = find_file(summary_path)
        tag_name = ['fast_rcnn_roi/none_zero_boxes', 'fast_rcnn_roi/none_zero_features', 'fast_rcnn_roi/rois_level']
        load_and_see_tensor(summary_path, tag_name)
    elif summary_name == 'resnet_v1_50_pascal_debug_fast_rcnn_predict':
        summary_path = find_file(summary_path)
        tag_name = ['fast_rcnn_net/summary_fast_rcnn_encode_boxes', 'fast_rcnn_net/summary_fast_rcnn_cls_scores']
        load_and_see_tensor(summary_path, tag_name)
    elif summary_name == 'resnet_v1_50_pascal_debug_fast_rcnn_predict_boxes':
        summary_path = find_file(summary_path)
        tag_name = ['fast_rcnn_prediction_boxes', 'image_shape']
        load_and_see_tensor(summary_path, tag_name)
    elif summary_name == 'resnet_v1_50_pascal_debug_fast_rcnn_match_proposal_function':
        summary_path = find_file(summary_path)
        tag_name = ['proposal_matched_boxes', 'proposal_matched_label', 'object_mask', 'max_iou_per_proposal']
        load_and_see_tensor(summary_path, tag_name)
    else:
        print('tensor name not included!!!')
