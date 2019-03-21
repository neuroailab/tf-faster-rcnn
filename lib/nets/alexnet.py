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
  def __init__(self, with_dropout=False):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'alexnet_v2'
    self.with_dropout = with_dropout

  def _image_to_head(self, is_training, reuse=None):
    batch_enabled = cfg.TRAIN.ENABLE_BATCH_NORM and is_training
    batch_norm_kwargs = {
        'momentum': 0.997, 
        'epsilon': 1e-5, 
        'training': batch_enabled, 
        'fused': True,
        }
    # BGR order std
    imagenet_std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    input_image = self._image / 255 / imagenet_std
    #input_image = tf.Print(input_image, [tf.shape(input_image)], message='Image shape', summarize=10)
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      net = slim.conv2d(input_image, 64, [11, 11], 4, padding='SAME',
                        scope='conv1', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv1', **batch_norm_kwargs)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1', padding='SAME')

      net = slim.conv2d(net, 192, [5, 5], scope='conv2', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=net,
              name='conv2', **batch_norm_kwargs)
      net = tf.nn.relu(net)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2', padding='SAME')

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

    #net = tf.Print(net, [tf.shape(net)], message='Net shape', summarize=10)
    self._act_summaries.append(net)
    self._layers['head'] = net
    return net

  def _head_to_tail(self, pool5, is_training, reuse=None):
    batch_enabled = cfg.TRAIN.ENABLE_BATCH_NORM and is_training
    batch_norm_kwargs = {
        'momentum': 0.997, 
        'epsilon': 1e-5, 
        'training': batch_enabled, 
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
      if self.with_dropout and is_training:
        net = slim.dropout(net, keep_prob=0.5, is_training=True, 
                            scope='dropout6')

      fc7 = slim.fully_connected(
          net, 4096, scope='fc7', activation_fn=None)
      net = tf.layers.batch_normalization(
              inputs=fc7,
              name='fc7', **batch_norm_kwargs)
      net = tf.nn.relu(net)
      if self.with_dropout and is_training:
        net = slim.dropout(net, keep_prob=0.5, is_training=True, 
                            scope='dropout7')
    return net

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic and 'Momentum' not in v.name:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix AlexNet layers..')
    with tf.variable_scope('Fix_AlexNet') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [5, 5, 256, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [11, 11, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
