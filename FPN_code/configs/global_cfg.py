# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35

import os
import sys
sys.path.append('../')


#####################
# All kinds of path #
#####################
# repository root path
ROOT_PATH = os.path.abspath("F:\\tensorflow\\detections\\my_FPN_Tensorflow")
# outer_path
OUTER_PATH = os.path.join(ROOT_PATH, 'outer_path')
# data set name
DATASET_NAME = 'pascal'
# network name
NETWORK_NAME = 'resnet_v1_50'
# version
VERSION = 'debug_fast_rcnn_match_proposal_function'
# summary path
SUMMARY_PATH = os.path.join(OUTER_PATH, 'output', 'summary', '%s_%s_%s'%(NETWORK_NAME, DATASET_NAME, VERSION))
# backbone network pretrain path
PRETRAIN_PATH = os.path.join(OUTER_PATH, 'pretrained_weight', NETWORK_NAME, NETWORK_NAME + '.ckpt')


##########################
# data process parameter #
##########################
# shortside lenght
SHORTSIDE_LEN = 800
# image depth mean value
DEPTH_MEAN = [103.939, 116.779, 123.68]
# batch_size
BATCH_SIZE = 1
# epoch
EPOCH = 20
# num classes
NUM_CLASSES = 20

#####################
# network parameter #
#####################
# -------backbone network-----------#
# weight decay
WEIGHT_DECAY = 1e-4

# -------rpn network---------------#
# anchor scales
ANCHOR_SCALES = [1]
# anchor ratios
ANCHOR_RATIOS = [1, 2, 0.5]
# scale_factor
SCALE_FACTOR = [1, 1, 1, 1]
# base_anchor_size_list
BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]
# stride
STRIDE = [4, 8, 16, 32, 64]
# level
LEVEL = ['P2', 'P3', 'P4', 'P5', 'P6']
# top k nms
TOP_K_NMS = 12000
# rpn_nms_iou_threshold
RPN_NMS_IOU_THRESHOLD = 0.5
# max number of proposal
MAX_PROPOSAL_NUM = 2000
# mini batch size of rpn
RPN_MINI_BATCH_SIZE = 512
# ratio of positive sample in mini batch
POSITIVE_RATIO = 0.5
# whether filter of outside anchors
IS_FILTER_OUTSIDE_ANCHORS = False
# iou threshold of positive sample of rpn net
RPN_IOU_POSITIVE_THRESHOLD = 0.7
# iou threshold of negative sample of rpn net
RPN_IOU_NEGATIVE_THRESHOLD = 0.3
# rpn net weight decay
RPN_WEIGHT_DECAY = 1e-4
# whether share head
IS_SHARE_HEAD = True

# -----fast rcnn net---------

# crop size
CROP_SIZE = 14
# roi_pooling_kernel_size
ROI_POOLING_KERNEL_SIZE = 2
# fast rcnn weights decay
FAST_RCNN_WEIGHTS_DECAY = 1e-4
# fast_rcnn_nms_iou_threshold
FAST_RCNN_NMS_IOU_THRESHOLD = 0.5
# max_num_per_class
MAX_NUM_PER_CLASS = 100
# fast_rcnn_score_threshold
FAST_RCNN_SCORE_THRESHOLD = 0.5
# fast_rcnn_positive_threshold_iou
FAST_RCNN_POSITIVE_THRESHOLD_IOU = 0.5

if __name__ == '__main__':
    # test this cfg.py
    print(NUM_CLASSES)
