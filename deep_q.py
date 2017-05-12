'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque 
import atari_game_wrapper as atari
import random

MODEL_ID = 'Atari-0'

# HyperParameter
DISCOUNT = 0.3
REPLAY_MEMORY = 500
BATCH_SIZE = 32
N_EPISODES = 100
BEFORE_TRAIN = 100#00
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
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
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, N_ACTIONS, sess, N_EPISODES, DISCOUNT):
        # init replay memory
        self.sess = sess
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_CHANNEL = IMG_CHANNEL
        self.N_ACTIONS = N_ACTIONS
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        self.replay_memory = deque()

    def approx_Q_network(self):
        x = tf.placeholder('float', [None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNEL])
        conv1 = conv(x, out_channel=32, kernel_size=8, stide_size=4)
        pool1 = maxpool(conv1, kernel_size=2, stide_size=2)
        conv2 = conv(pool1, out_channel=64, kernel_size=4, stide_size=2)
        pool2 = maxpool(conv2, kernel_size=2, stide_size=2)
        conv3 = conv(pool2, out_channel=64, kernel_size=3, stide_size=1)
        pool3 = maxpool(conv3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        fc1 = fully_connect(flatten, 256)
        output_Q = fully_connect(fc1, self.N_ACTIONS)
        return x, output_Q

    def train_network(self):
        # Define cost function of network
        x, output_Q = self.approx_Q_network()  # output_Q: (batch, N_ACTIONS)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])
        y = tf.placeholder("float", [None])
        cost = tf.reduce_mean(tf.square(y - max_action_Q))
        train_step = tf.train.AdamOptimizer(0.001).minimize(cost)

        # emulate and store trainsitions into replay_memory
        game = atari.Game('AirRaid-v0')
        state_t = game.initial_state()
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        for episode in range(self.N_EPISODES):
            for t in range(1000000000):
                # Select one action
                if(self.explore()):
                    action_t = game.random_action()
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1,80,80,4))})[0]
                # Execute the chosen action in emulator
                state_t1, reward_t, terminal, info = game.step(action_t)
                self.store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal)
                state_t = state_t1
                if terminal:
                    state_t = game.initial_state()
                    break
                # Train the approx_Q_network
                if self.global_time > BEFORE_TRAIN:
                    transition_batch = random.sample(self.replay_memory, BATCH_SIZE)
                    transition_batch = np.array(transition_batch)
                    print('transition_batch: ', transition_batch.shape)
                    state_j, action_j, reward_j, state_j1, terminal_j1  = np.split(transition_batch, 5, axis=1)
                    print('state_j:', state_j.shape)
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * max_action_Q.eval(feed_dict={x: state_j1})[0] )
                    print('y_j:', y_j.shape)
                    train_step.run(feed_dict={x:state_j, y:y_j})
                #if t % 1000 == 0:
                   #saver.save(sess, 'models/' + MODEL_ID, global_step = t)
            self.global_time += t
            print('Episode {:3d}: {:6d}'.format(episode, t))
            saver.save(sess, 'models/' + MODEL_ID)

    def explore(self):
        if(self.global_time <= BEFORE_TRAIN):
            return True
        elif self.epsilon > FINAL_EPSILON and self.global_time % 1000 == 0:
            self.epsilon *= 0.9978
            print('new epsilon = {:.4f}'.format(self.epsilon))
            return random.random() < self.epsilon

    def store_to_replay_memory(self, state_t, action_t, reward_t, state_t1, terminal):
        transition = [state_t, action_t, reward_t, state_t1, terminal]
        self.replay_memory.append(transition)
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(80, 80, 4, 6, sess, N_EPISODES, DISCOUNT)
    dqn.train_network()

