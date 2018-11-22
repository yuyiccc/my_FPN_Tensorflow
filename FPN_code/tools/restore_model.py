# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/21 21:38

import tensorflow.contrib.slim as slim
import sys
sys.path.append('../')
import configs.global_cfg as cfg
import tensorflow as tf


def get_restorer():

    print('pretrained backbone mode path is :%s' % cfg.PRETRAIN_PATH)

    model_variables = slim.get_model_variables()

    restore_variables = [var for var in model_variables if
                         (var.name.startswith(cfg.NETWORK_NAME)
                         and not var.name.startswith('{}/logits'.format(cfg.NETWORK_NAME)))]
    for var in restore_variables:
        print(var.name)
    restorer = tf.train.Saver(restore_variables)

    return cfg.PRETRAIN_PATH, restorer



