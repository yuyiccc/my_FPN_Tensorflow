# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35

import os

#####################
# All kinds of path #
#####################
# repository root path
ROOT_PATH = os.path.abspath("F:\\tensorflow\\detections\\my_FPN_Tensorflow\\")
# VOC path
VOC_PATH = os.path.join(ROOT_PATH, 'outer_path', 'VOCdevkit','VOC2012')
# detect dataset txt pth
DATASET_TXT_PATH = os.path.join(VOC_PATH, 'ImageSets', 'Main', 'trainval.txt')
# images path
IM_PATH = os.path.join(VOC_PATH, 'JPEGImages')
# xml path
XML_PATH = os.path.join(VOC_PATH, 'Annotations')
# source dataset path
SOURCE_DATA_PATH = os.path.join(ROOT_PATH, 'outer_path', 'source_data_set')
# source images path
SOURCE_IM_PATH = os.path.join(SOURCE_DATA_PATH, 'IMAGES')
# source xml path
SOURCE_XML_PATH = os.path.join(SOURCE_DATA_PATH, 'XML')


if __name__=='__main__':
    # test this cfg.py

    # test path
    import sys
    sys.path.append('../')
    from tools.assist_tools import check_paths_or_files

    check_tuple = (ROOT_PATH, VOC_PATH, DATASET_TXT_PATH, IM_PATH, XML_PATH,
                   SOURCE_DATA_PATH, SOURCE_IM_PATH, SOURCE_XML_PATH)
    check_paths_or_files(check_tuple)



