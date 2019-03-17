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
import pdb

class vgg16_inst(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._scope = 'vgg_16'

  def _image_to_head(self, is_training, reuse=None):
    normalizer_fn = tf.layers.batch_normalization
    normalizer_params = {
        'momentum': 0.997,
        'epsilon': 1e-5,
        'training': is_training,
        'fused': True}
    kwargs = {
        'normalizer_fn': normalizer_fn,
        'normalizer_params': normalizer_params}

    # BGR order std
    imagenet_std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    input_image = self._image / 255 / imagenet_std
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      '''
      net = slim.repeat(
              inputs, 1, slim.conv2d, 64, [3, 3], 
              trainable=False, scope='conv1')
      '''
      net = slim.repeat(
              input_image, 1, slim.conv2d, 64, [3, 3], trainable=False, 
              scope='conv1_final', **kwargs)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
      net = slim.repeat(
              net, 1, slim.conv2d, 128, [3, 3], 
              trainable=False, scope='conv2')
      net = slim.repeat(
              net, 1, slim.conv2d, 128, [3, 3], trainable=False, 
              scope='conv2_final', **kwargs)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
      net = slim.repeat(
              net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.repeat(
              net, 1, slim.conv2d, 256, [3, 3], scope='conv3_final', **kwargs)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
      net = slim.repeat(
              net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.repeat(
              net, 1, slim.conv2d, 512, [3, 3], scope='conv4_final', **kwargs)
      net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
      net = slim.repeat(
              net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.repeat(
              net, 1, slim.conv2d, 512, [3, 3], scope='conv5_final', **kwargs)
      #net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool5')

    self._act_summaries.append(net)
    self._layers['head'] = net
    
    return net

  def _head_to_tail(self, pool5, is_training, reuse=None):
    normalizer_fn = tf.layers.batch_normalization
    normalizer_params = {
        'momentum': 0.997,
        'epsilon': 1e-5,
        'training': is_training,
        'fused': True}
    kwargs = {
        'normalizer_fn': normalizer_fn,
        'normalizer_params': normalizer_params}
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      pool5_flat = slim.flatten(pool5, scope='flatten')
      fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6', **kwargs)
      fc7 = slim.fully_connected(fc6, 4096, scope='fc7', **kwargs)

    return fc7

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the conv weights that are fc weights in vgg16
      if v.name == (self._scope + '/fc6/weights:0') or \
         v.name == (self._scope + '/fc7/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      # exclude the first conv layer to swap RGB to BGR
      if v.name == (self._scope + '/conv1_final/conv1_final_1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic and 'Momentum' not in v.name:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix VGG16 layers..')
    with tf.variable_scope('Fix_VGG16') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        fc6_conv = tf.get_variable("fc6_conv", [7, 7, 512, 4096], trainable=False)
        fc7_conv = tf.get_variable("fc7_conv", [1, 1, 4096, 4096], trainable=False)
        conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
        restorer_fc = tf.train.Saver({self._scope + "/fc6/weights": fc6_conv, 
                                      self._scope + "/fc7/weights": fc7_conv,
                                      self._scope + "/conv1_final/conv1_final_1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc6/weights:0'], tf.reshape(fc6_conv, 
                            self._variables_to_fix[self._scope + '/fc6/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/fc7/weights:0'], tf.reshape(fc7_conv, 
                            self._variables_to_fix[self._scope + '/fc7/weights:0'].get_shape())))
        sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1_final/conv1_final_1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
