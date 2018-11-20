# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:22
import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append("../../")
import configs.global_cfg as cfg




def draw_box_with_tensor(img_batch, boxes, text):
    '''

    :param img_batch: [1,h,w,3]
    :param boxes: [-1,5]
    :param text: image name
    :return: img_tensor_with_boxes same size as img_batch
    '''
    def draw_box_cv(img, boxes, text):
        img += cfg.DEPTH_MEAN
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

        text = text.decode()
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        # img = np.transpose(img, [2, 1, 0])
        img = img[:, :, -1::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text[0]],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes