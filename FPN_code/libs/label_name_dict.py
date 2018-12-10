# -*- coding: utf-8 -*-
# @Time    : 2018/11/17 12:04
# @Author  : YuYi
import sys
sys.path.append('../')
import configs.global_cfg as cfg


if cfg.DATASET_NAME == 'pascal':
    LABEL_TO_NUMBER = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
    NUMBER_TO_LABEL = {LABEL_TO_NUMBER[key]: key for key in LABEL_TO_NUMBER}

if __name__=='__main__':
    print(LABEL_TO_NUMBER)
    print(NUMBER_TO_LABEL)
