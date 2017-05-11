import tensorflow as tf
import numpy as np

# Parameter
discount = 0.3

''' tf Wrapper '''
def add_conv(in_tensor, out_size, conv_filter):
  biases_initializer = tf.constant_initializer(0.)
  weights_initializer = tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32)

  conv = tf.contrib.layers.conv2d(in_tensor, out_size, kernel_size=conv_filter, stride=params['conv_stride'], activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, padding='SAME')
  return tf.nn.relu(conv)

def add_maxpool(in_tensor):
  pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=params['pool_kernel'], stride=params['pool_stride'], padding='SAME')
  return pool

def add_lrn(in_tensor):
  return tf.nn.lrn(in_tensor, depth_radius=5, bias=1.0, alpha=0.001/9.0, beta=0.75)

def add_fully(in_tensor, n_out):
  biases_initializer = tf.constant_initializer(0.)
  weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
  fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
  return tf.nn.relu(fc)

