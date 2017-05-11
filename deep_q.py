'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque 
import atari_game_wrapper as game

# HyperParameter
DISCOUNT = 0.3
REPLAY_MEMORY = 50000
BATCH_SIZE = 32
# simulated annealing

# state = (img_width, img_height, num_history_frames)

''' tf Wrapper '''
def conv(in_tensor, out_channel, kernel_size, stide_size):
    biases_initializer = tf.constant_initializer(0.01)
    weights_initializer = tf.truncated_normal_initializer(mean=0, stddev = 0.01)
    conv = tf.contrib.layers.conv2d(in_tensor, out_channel, kernel_size=kernel_size, stride=stide_size, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, padding='SAME')
    return tf.nn.relu(conv)

def maxpool(in_tensor, kernel_size, stide_size):
    pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=kernel_size, stride=stide_size, padding='SAME')
    return pool

def lrn(in_tensor):
    return tf.nn.lrn(in_tensor, depth_radius=5, bias=1.0, alpha=0.001/9.0, beta=0.75)

def fully_connect(in_tensor, n_out):
    biases_initializer = tf.constant_initializer(0.01)
    weights_initializer = tf.truncated_normal_initializer(mean=0, stddev = 0.01)
    fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
    return tf.nn.relu(fc)


class DeepQ():
    def __init__(self, IMG_WIDTH=80, IMG_HEIGHT=80, IMG_CHANNEL=4, N_ACTIONS ):
        # init replay memory
        self.replayMemory = deque()

    def create_network(self):
        x = tf.placeholder('float', [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL])
        conv1 = conv(x, out_channel=32, kernel_size=8, stide_size=4)
        pool1 = maxpool(conv1, kernel_size=2, stide_size=2)
        conv2 = conv(pool1, out_channel=64, kernel_size=4, stide_size=2)
        pool2 = maxpool(conv2, kernel_size=2, stide_size=2)
        conv3 = conv(pool2, out_channel=64, kernel_size=3, stide_size=1)
        pool3 = maxpool(conv3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        fc1 = fully_connect(flatten, 256)
        output_Q = fully_connect(fc1, N_ACTIONS)
        return x, output_Q


    def train_network(self):
        x, output_Q = create_network()
        y_Q = tf.placeholder("float", [None]) # Q-value target

    




