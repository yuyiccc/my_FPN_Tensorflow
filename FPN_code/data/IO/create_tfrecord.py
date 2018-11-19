# -*- coding: utf-8 -*-
# @Time    : 2018/11/16 22:25
# @Author  : YuYi
import tensorflow as tf
import sys
sys.path.append('../../')
import configs.global_cfg as cfg
import os
from tools.assist_tools import ShowProcess
import argparse
import cv2
import  xml.etree.cElementTree as ET
from libs.label_name_dict import LABEL_TO_NUMBER
import numpy as np
from tools.assist_tools import check_and_create_paths
parser = argparse.ArgumentParser()
parser.add_argument('--im_type', default='.jpg', type=str, help='image type')
parser.add_argument('--dataset_name', default='train_dataset', type=str, help='dataset name')
parser.add_argument('--records_name', default='train.tfrecord', type=str, help='tfrecord name')
args = parser.parse_args()
# dataset path
DATA_PATH = os.path.join(cfg.OUTER_PATH, 'source_data_set', cfg.DATASET_NAME, args.dataset_name)
IM_PATH = os.path.join(DATA_PATH, 'image_files')
XML_PATH = os.path.join(DATA_PATH, 'xml_files')
# tfrecord path
TFRECORD_PATH = os.path.join(cfg.OUTER_PATH, 'tfrecords', cfg.DATASET_NAME)
check_and_create_paths([TFRECORD_PATH])
TFRECORD_PATH = os.path.join(TFRECORD_PATH, args.records_name)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml(xml_path):
    tree = ET.parse(xml_path)

    im_size = tree.findall('size')[0]
    img_height = int(im_size.find('height').text)
    img_width = int(im_size.find('width').text)

    objs = tree.findall('object')
    boxes = []
    for obj in objs:
        obj_name = obj.find('name').text
        label = LABEL_TO_NUMBER[obj_name]
        assert label is not None, 'this %s is not in dict!!!' % obj_name

        box = obj.find('bndbox')
        xmin = box.find('xmin').text
        ymin = box.find('ymin').text
        xmax = box.find('xmax').text
        ymax = box.find('ymax').text

        boxes.append([ymin, xmin, ymax, xmax, label])
    gtbox_label = np.array(boxes, dtype=np.int32)

    return img_height, img_width, gtbox_label
# check xml's image size is the same as images,and all box are in the image.
def check_image_and_boxes(im_shape,xml_shape,boxes):
    # check image size
    assert (im_shape == xml_shape).all(), \
        "im's shape is not equal to xml' image shape. im:(%d,%d) xml:(%d,%d)"\
        %(im_shape[0], im_shape[1], xml_shape[0], xml_shape[1])

    # check boxes
    assert (boxes > 0).all(), "gt_boxes coord less 0"
    assert (boxes[:, ::2] <= im_shape[0]).all(), "gt_boxes' y greater than height"
    assert (boxes[:, 1::2] <= im_shape[1]).all(), "gt_boxes' x greater than weight"
    assert (boxes[:, 0] < boxes[:, 2]), "xmin > xmax "
    assert (boxes[:, 1] < boxes[:, 3]), "ymin > ymax "


def convert_data_to_tfrecord():
    im_list = os.listdir(IM_PATH)
    data_name_list = [im.split('.')[0] for im in im_list]
    num_data = len(data_name_list)
    process = ShowProcess(num_data)

    writer = tf.python_io.TFRecordWriter(TFRECORD_PATH)
    for name_i in data_name_list:
        process.show_process()
        xml_i_path = os.path.join(XML_PATH, name_i+'.xml')
        image_i_path = os.path.join(IM_PATH, name_i+args.im_type)

        img_height, img_width, gtbox_label = read_xml(xml_i_path)
        img = cv2.imread(image_i_path)

        # check image and boxes
        xml_shape = np.array([img_height, img_width])
        im_shape = np.array(img.shape[:2])
        check_image_and_boxes(im_shape=im_shape, xml_shape=xml_shape, boxes=gtbox_label[:, :4])

        features = tf.train.Features(feature={
            'img_name': _bytes_feature(name_i.encode()),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtbox_label': _bytes_feature(gtbox_label.tostring()),
            'num_gtbox': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()
    print('Conversion is Done!!!')


if __name__=='__main__':
    convert_data_to_tfrecord()

