# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/11/20 20:37
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
sys.path.append('../../')
import configs.global_cfg as cfg
from libs.networks.nets import resnet_v1


def get_network_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       spatial_squeeze=True):
    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.WEIGHT_DECAY)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        num_classes=num_classes,
                                                        is_training=is_training,
                                                        global_pool=global_pool,
                                                        output_stride=output_stride,
                                                        spatial_squeeze=spatial_squeeze
                                                        )

        return logits, end_points
    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.WEIGHT_DECAY)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         spatial_squeeze=spatial_squeeze
                                                         )
        return logits, end_points


if __name__ == '__main__':
    # debug the feature map's stride problem
    image_shape = [1, 600, 917, 3]
    image = tf.random_normal(image_shape)
    _, end_point = get_network_byname(net_name=cfg.NETWORK_NAME,
                                      inputs=image,
                                      num_classes=None,
                                      is_training=True,
                                      global_pool=False,
                                      output_stride=None,
                                      spatial_squeeze=False)
    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    with tf.Session() as sess:
        sess.run(init_op)
        feature_map = sess.run(end_point)
        for key in feature_map.keys():
            print(key, ' shape:', feature_map[key].shape)






