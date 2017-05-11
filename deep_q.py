'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque 
import atari_game_wrapper as game

# Parameter
DISCOUNT = 0.3
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
# simulated annealing

class DeepQ():
    def __init__(self):
        # init replay memory
        self.replayMemory = deque()



    ''' tf Wrapper '''
    def conv(in_tensor, out_size, filter_size, stide_size):
        biases_initializer = tf.constant_initializer(0.)
        weights_initializer = tf.uniform_unit_scaling_initializer(seed=None, dtype=tf.float32)
        conv = tf.contrib.layers.conv2d(in_tensor, out_size, kernel_size=filter_size, stride=stide_size, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, padding='SAME')
        return tf.nn.relu(conv)

    def maxpool(in_tensor):
        pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=params['pool_kernel'], stride=params['pool_stride'], padding='SAME')
        return pool

    def lrn(in_tensor):
        return tf.nn.lrn(in_tensor, depth_radius=5, bias=1.0, alpha=0.001/9.0, beta=0.75)

    def fully_connect(in_tensor, n_out):
        biases_initializer = tf.constant_initializer(0.)
        weights_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.02, seed=None, dtype=tf.float32)
        fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
        return tf.nn.relu(fc)
