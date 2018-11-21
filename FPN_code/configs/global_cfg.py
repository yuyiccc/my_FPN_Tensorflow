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
VERSION = 'v1'
# summary path
SUMMARY_PATH = os.path.join(OUTER_PATH, 'output', 'summary', '%s_%s_%s'%(NETWORK_NAME, DATASET_NAME, VERSION))
# backbone network pretrain path
PRETRAIN_PATH = os.path.join(OUTER_PATH, 'pretrained_weight', NETWORK_NAME)



##########################
# data process parameter #
##########################
# shortside lenght
SHORTSIDE_LEN = 600
# image depth mean value
DEPTH_MEAN = [103.939, 116.779, 123.68]
# batch_size
BATCH_SIZE = 1
# epoch
EPOCH = 20
# num classes
NUM_CLASSES = 21

#####################
# network parameter #
#####################
# weight decay
WEIGHT_DECAY = 1e-4
#




if __name__=='__main__':
    # test this cfg.py
    print(NUM_CLASSES)
