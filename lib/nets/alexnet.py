from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


class alexnet(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'alexnet_v2'

  def _image_to_head(self, is_training, reuse=None):
    batch_norm_kwargs = {
        'momentum': 0.997, 
        'epsilon': 1e-5, 
        'training': False, 
        'fused': True,
        }
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net = slim.conv2d(self._image, 64, [11, 11], 4, padding='VALID',
                        scope='conv1', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv1', **batch_norm_kwargs)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')

      net = slim.conv2d(net, 192, [5, 5], scope='conv2', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv2', **batch_norm_kwargs)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')

      net = slim.conv2d(net, 384, [3, 3], scope='conv3', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv3', **batch_norm_kwargs)
      net = tf.nn.relu(net)

      net = slim.conv2d(net, 384, [3, 3], scope='conv4', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv4', **batch_norm_kwargs)
      net = tf.nn.relu(net)

      net = slim.conv2d(net, 256, [3, 3], scope='conv5', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv5', **batch_norm_kwargs)
      net = tf.nn.relu(net)

    self._act_summaries.append(net)
    self._layers['head'] = net
    return net

  def _head_to_tail(self, pool5, is_training, reuse=None):
    batch_norm_kwargs = {
        'momentum': 0.997, 
        'epsilon': 1e-5, 
        'training': False, 
        'fused': True,
        }
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(
          pool5_flat, 4096, scope='fc6', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=fc6,
              name='fc6', **batch_norm_kwargs)
      net = tf.nn.relu(net)

      fc7 = slim.fully_connected(
          net, 4096, scope='fc7', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=fc7,
              name='fc7', **batch_norm_kwargs)
      net = tf.nn.relu(net)
    return fc7
