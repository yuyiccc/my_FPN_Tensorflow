# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35

import os

#####################
# All kinds of path #
#####################
# repository root path
ROOT_PATH = os.path.abspath("G:\\tensorflow\\FPN\\my_FPN_Tensorflow")
# outer_path
OUTER_PATH = os.path.join(ROOT_PATH, 'outer_path')
# data set name
DATASET_NAME = 'pascal'

if __name__=='__main__':
    # test this cfg.py
    print(1)
