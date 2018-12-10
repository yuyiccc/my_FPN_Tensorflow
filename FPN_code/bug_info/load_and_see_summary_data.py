# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/6 15:17

import tensorflow as tf
import numpy as np

tensor_name = 'fast_rcnn_boxes_prediction'


def load_and_see_tensor(summary_path, tag_name):
    for e in tf.train.summary_iterator(summary_path):
        for v in e.summary.value:
            if v.tag in tag_name:
                fb = np.frombuffer(v.tensor.tensor_content, dtype=np.float32)
                shape = []
                for d in v.tensor.tensor_shape.dim:
                    shape.append(d.size)
                fb = fb.reshape(shape)
                print(fb)
                print(v.tag)


if tensor_name == 'location_l1_loss':
    summary_path = "F:\\tensorflow\\detections\\my_FPN_Tensorflow\\outer_path\\output\\summary\\resnet_v1_50_pascal_debug_rpn_loss_function\\events.out.tfevents.1544079074.WIN-RR5RDDKFI20"
    tag_name = ['rpn_loss/rpn_localization_losses/tensor/location_l1_loss']
    load_and_see_tensor(summary_path, tag_name)
elif tensor_name == 'none_zero_rois':
    summary_path = "F:\\tensorflow\\detections\\my_FPN_Tensorflow\\outer_path\\output\\summary\\resnet_v1_50_pascal_debug_get_rois\\events.out.tfevents.1544408081.WIN-RR5RDDKFI20"
    tag_name = ['fast_rcnn_roi/none_zero_boxes', 'fast_rcnn_roi/none_zero_features', 'fast_rcnn_roi/rois_level']
    load_and_see_tensor(summary_path, tag_name)
elif tensor_name == 'fast_rcnn_predict':
    summary_path = "F:\\tensorflow\\detections\\my_FPN_Tensorflow\\outer_path\\output\\summary\\resnet_v1_50_pascal_debug_fast_rcnn_predict\\events.out.tfevents.1544418468.WIN-RR5RDDKFI20"
    tag_name = ['fast_rcnn_net/summary_fast_rcnn_encode_boxes', 'fast_rcnn_net/summary_fast_rcnn_cls_scores']
    load_and_see_tensor(summary_path, tag_name)
elif tensor_name == 'fast_rcnn_boxes_prediction':
    summary_path = "F:\\tensorflow\\detections\\my_FPN_Tensorflow\\outer_path\\output\\summary\\resnet_v1_50_pascal_debug_fast_rcnn_predict_boxes\\events.out.tfevents.1544449905.WIN-RR5RDDKFI20"
    tag_name = ['fast_rcnn_prediction_boxes', 'image_shape']
    load_and_see_tensor(summary_path, tag_name)
else:
    print('tensor name not included!!!')
