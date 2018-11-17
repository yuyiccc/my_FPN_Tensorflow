# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35
import shutil
import os
import sys
sys.path.append('../../')
from  configs import global_cfg as cfg
from tools.assist_tools import *

# path and file
# VOC path
VOC_PATH = os.path.join(cfg.ROOT_PATH, 'outer_path', 'VOCdevkit', 'VOC2012')
# detect dataset txt pth
DATASET_TXT_PATH = os.path.join(VOC_PATH, 'ImageSets', 'Main', 'trainval.txt')
# images path
IM_PATH = os.path.join(VOC_PATH, 'JPEGImages')
# xml path
XML_PATH = os.path.join(VOC_PATH, 'Annotations')
# source dataset path
SOURCE_DATA_PATH = os.path.join(cfg.ROOT_PATH, 'outer_path', 'source_data_set')
# source images path
SOURCE_IM_PATH = os.path.join(SOURCE_DATA_PATH, 'IMAGES')
# source xml path
SOURCE_XML_PATH = os.path.join(SOURCE_DATA_PATH, 'XML')



# check files and path
paths = (IM_PATH, XML_PATH,
        SOURCE_DATA_PATH, SOURCE_IM_PATH, SOURCE_XML_PATH)
check_and_create_paths(paths)

files = (DATASET_TXT_PATH, DATASET_TXT_PATH)
check_files(files)



data_name_list  = open(DATASET_TXT_PATH).readlines()
max_len = len(data_name_list)
process_bar = ShowProcess(max_len)
for name_i in data_name_list:
    # erase /n
    process_bar.show_process()
    name_i = name_i[:-1]
    im_file = os.path.join(IM_PATH, name_i+'.jpg')
    xml_file = os.path.join(XML_PATH, name_i+'.xml')
    out_im_file = os.path.join(SOURCE_IM_PATH, name_i+'.jpg')
    out_xml_file = os.path.join(SOURCE_XML_PATH, name_i+'.xml')
    shutil.copyfile(im_file, out_im_file)
    shutil.copyfile(xml_file, out_xml_file)

