
# -*- coding: utf-8 -*-
# Author:YuYi
# Time :2018/12/6 20:06

import tensorflow as tf


a = tf.constant([1,2,3,5,6,4,2,1])
b = []
for i in range(8):
    if a[i]>4:
        continue
    b.append(a[i])
sess = tf.Session()
print(sess.run(b))
