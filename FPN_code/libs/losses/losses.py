# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/4 21:56
import tensorflow as tf


def l1_smooth_losses(predict_boxes,
                     gtboxes,
                     object_weights,
                     classes_weights=None):
    '''
    :param predict_boxes: [N, 4]
    :param gtboxes: [N, 4]
    :param object_weights: [N,]
    :param classes_weights: [N, 4*num_cls]
    :return: 
    '''

    diff = tf.cast(tf.abs(predict_boxes-gtboxes), dtype=tf.float32)
    object_weights = tf.cast(object_weights, dtype=tf.float32)
    if classes_weights is not None:
        classes_weights = tf.cast(classes_weights, dtype=tf.float32)

    if classes_weights is None:
        boxes_smooth_l1_losses = tf.reduce_sum(
            tf.where(tf.less(diff, 1), 0.5*tf.square(diff), diff-0.5), axis=1)*object_weights
        # tf.summary.tensor_summary('tensor/location_l1_loss', boxes_smooth_l1_losses)
    else:
        boxes_smooth_l1_losses = tf.reduce_sum(
            tf.where(tf.less(diff, 1),
                     0.5*tf.square(diff)*classes_weights,
                     (diff-0.5)*classes_weights)
        )*object_weights

    num_object = tf.shape(tf.where(tf.equal(object_weights, 1)))[0]
    losses = tf.cond(tf.equal(num_object, 0),
                     lambda: tf.constant(0, dtype=tf.float32),
                     lambda: tf.reduce_sum(boxes_smooth_l1_losses, axis=0)/tf.cast(num_object, tf.float32)
                     )

    return losses


if __name__ == '__main__':
    predict = tf.constant([[0, 0, 10, 10], [10, 10, 15, 15]], dtype=tf.float32)
    label = tf.constant([[1, 1, 2, 2], [3, 3, 4, 5]], dtype=tf.float32)
    object_mask = tf.constant([1, 1], dtype=tf.float32)
    loss = l1_smooth_losses(predict, label, object_mask)
    with tf.Session() as sess:
        l_a = sess.run(loss)
        print(l_a)


