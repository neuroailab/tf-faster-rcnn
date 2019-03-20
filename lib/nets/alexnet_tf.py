from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np
from tfutils.model_tool import ConvNet

from nets.network import Network
from model.config import cfg


class alexnet_tf(Network):
  def __init__(self):
    Network.__init__(self)
    self._feat_stride = [16, ]
    self._feat_compress = [1. / float(self._feat_stride[0]), ]
    self._m = ConvNet(seed=0)
    self._scope = 'alexnet_tf'

  def _image_to_head(self, is_training, reuse=None):
    # BGR order std
    imagenet_std = np.array([0.225, 0.224, 0.229], dtype=np.float32)
    input_image = self._image / 255 / imagenet_std

    m = self._m
    # Define model class and default kwargs for different types of layers
    conv_kwargs = {
            'add_bn': False,
            'init': 'xavier',
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            }
    pool_kwargs = {
            'pool_type': 'maxpool',
            }

    # Actually define the network
    m.conv(
            96, 11, 4, padding='SAME', 
            layer='conv1', in_layer=input_image, **conv_kwargs)
    m.pool(3, 2, **pool_kwargs)

    m.conv(256, 5, 1, layer='conv2', **conv_kwargs)
    m.pool(3, 2, **pool_kwargs)

    m.conv(384, 3, 1, layer='conv3', **conv_kwargs)
    m.conv(384, 3, 1, layer='conv4', **conv_kwargs)

    net = m.conv(256, 3, 1, layer='conv5', **conv_kwargs)

    self._act_summaries.append(net)
    self._layers['head'] = net
    return net

  def _head_to_tail(self, pool5, is_training, reuse=None):
    fc_kwargs = {
            'init': 'trunc_norm',
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
            'stddev': .01,
            }
    m = self._m
    dropout = .5 if is_training else None
    m.fc(4096, in_layer=pool5, dropout=dropout, bias=.1, layer='fc6', **fc_kwargs)
    net = m.fc(4096, dropout=dropout, bias=.1, layer='fc7', **fc_kwargs)
    return net

  def get_variables_to_restore(self, variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
      # exclude the first conv layer to swap RGB to BGR
      if v.name == ('conv1/weights:0'):
        self._variables_to_fix[v.name] = v
        continue
      if v.name.split(':')[0] in var_keep_dic \
          and 'Momentum' not in v.name \
          and 'Variable:' not in v.name:
        print('Variables restored: %s' % v.name)
        variables_to_restore.append(v)

    return variables_to_restore

  def fix_variables(self, sess, pretrained_model):
    print('Fix AlexNet layers..')
    with tf.variable_scope('Fix_AlexNet') as scope:
      with tf.device("/cpu:0"):
        # fix the vgg16 issue from conv weights to fc weights
        # fix RGB to BGR
        conv1_rgb = tf.get_variable("conv1_rgb", [11, 11, 3, 96], trainable=False)
        restorer_fc = tf.train.Saver({"conv1/weights": conv1_rgb})
        restorer_fc.restore(sess, pretrained_model)

        sess.run(tf.assign(self._variables_to_fix['conv1/weights:0'], 
                            tf.reverse(conv1_rgb, [2])))
