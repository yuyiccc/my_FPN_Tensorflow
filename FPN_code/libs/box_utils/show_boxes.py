# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/19 20:22

def draw_box_with_tensor(img_tensor, gtboxes_label, text):
    '''
    :param img_tensor: [1,h,w,c]
    :param gtboxes_label: [-1,4]
    :param text: [-1]
    :return: gtboxes_in_img : []
    '''