# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/15 15:35
import shutil
import os
import sys
sys.path.append('../../')
from  configs import global_cfg as cfg
from tools.assist_tools import *


# check files and path
paths = (cfg.IM_PATH, cfg.XML_PATH,
        cfg.SOURCE_DATA_PATH, cfg.SOURCE_IM_PATH, cfg.SOURCE_XML_PATH)
check_and_create_paths(paths)

files = (cfg.DATASET_TXT_PATH, cfg.DATASET_TXT_PATH)
check_files(files)



data_name_list  = open(cfg.DATASET_TXT_PATH).readlines()
max_len = len(data_name_list)
process_bar = ShowProcess(max_len)
for name_i in data_name_list:
    # erase /n
    process_bar.show_process()
    name_i = name_i[:-1]
    im_file = os.path.join(cfg.IM_PATH, name_i+'.jpg')
    xml_file = os.path.join(cfg.XML_PATH, name_i+'.xml')
    out_im_file = os.path.join(cfg.SOURCE_IM_PATH, name_i+'.jpg')
    out_xml_file = os.path.join(cfg.SOURCE_XML_PATH, name_i+'.xml')
    shutil.copyfile(im_file, out_im_file)
    shutil.copyfile(xml_file, out_xml_file)

