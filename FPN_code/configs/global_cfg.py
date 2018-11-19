# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35

import os

#####################
# All kinds of path #
#####################
# repository root path
ROOT_PATH = os.path.abspath("F:\\tensorflow\\detections\\my_FPN_Tensorflow")
# outer_path
OUTER_PATH = os.path.join(ROOT_PATH, 'outer_path')
# data set name
DATASET_NAME = 'pascal'


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




if __name__=='__main__':
    # test this cfg.py
    print(1)
