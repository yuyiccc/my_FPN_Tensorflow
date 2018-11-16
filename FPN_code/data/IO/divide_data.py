# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/16 16:28

import shutil
import random
import os
import sys
sys.path.append('../../')
from configs import global_cfg as cfg
from tools.assist_tools import check_and_create_paths, ShowProcess
import  argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', default=0.8, type=float, help='number train data / number total data')
parser.add_argument('--im_type', default='.jpg', type=str, help="image's type")
args = parser.parse_args()




# source dataset path
SOURCE_DATA_PATH = os.path.join(cfg.ROOT_PATH, 'outer_path', 'source_data_set')
# source images path
SOURCE_IM_PATH = os.path.join(SOURCE_DATA_PATH, 'IMAGES')
# source xml path
SOURCE_XML_PATH = os.path.join(SOURCE_DATA_PATH, 'XML')

if not os.path.isdir(SOURCE_IM_PATH) and not os.path.isdir(SOURCE_XML_PATH):
    raise NameError('source data path is not right!!!')


def create_dataset_path(dataset_name = 'train'):
    dataset_path = os.path.join(SOURCE_DATA_PATH, '%s_dataset'%dataset_name)
    dataset_image_path = os.path.join(dataset_path, 'image_files')
    dataset_xml_path = os.path.join(dataset_path, 'xml_files')
    return (dataset_path,dataset_image_path,dataset_xml_path)


# train dataset path and test data path
TRAIN_DATASET_PATH, TRAIN_IM_PATH, TRAIN_XML_PATH = create_dataset_path('train')
# test dataset path and test data path
TEST_DATASET_PATH, TEST_IM_PATH, TEST_XML_PATH = create_dataset_path('test')

# create the path if it not exists
paths = (TRAIN_DATASET_PATH, TRAIN_IM_PATH, TRAIN_XML_PATH, TEST_DATASET_PATH, TEST_IM_PATH, TEST_XML_PATH)
check_and_create_paths(paths)

# read data name as a list
im_list = os.listdir(SOURCE_IM_PATH)
name_list = [im.split('.')[0] for im in im_list]
random.shuffle(name_list)
num_data = len(name_list)

# divide dataset
num_train_data = int(args.ratio * num_data)
train_list = name_list[:num_train_data]
test_list = name_list[num_train_data:]

# move data


def move_data(data_list, in_im_path, in_xml_path, out_im_path, out_xml_path):
    process = ShowProcess(len(data_list))
    for data_i in data_list:
        process.show_process()
        in_im_files = os.path.join(in_im_path, data_i+args.im_type)
        out_im_file = os.path.join(out_im_path, data_i+args.im_type)
        in_xml_files = os.path.join(in_xml_path, data_i+'.xml')
        out_xml_files = os.path.join(out_xml_path, data_i+'.xml')
        shutil.copy(in_im_files, out_im_file)
        shutil.copy(in_xml_files, out_xml_files)


move_data(train_list, SOURCE_IM_PATH, SOURCE_XML_PATH, TRAIN_IM_PATH, TRAIN_XML_PATH)
print('train dataset is Done!')
move_data(test_list, SOURCE_IM_PATH, SOURCE_XML_PATH, TEST_IM_PATH, TEST_XML_PATH)
print('test dataset is Done!')



