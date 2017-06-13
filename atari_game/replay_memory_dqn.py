'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque
import atari_game_wrapper as atari
import random
import os
import json
from memory import Memory
try:
    import resource
except:
    pass

MODEL_ID = 'replay-1'
directory = 'models/{}'.format(MODEL_ID)

# HyperParameter
SKIP_FRAMES = 1 #4
DISCOUNT = 0.99
LEARNING_RATE = 0.0001
REPLAY_MEMORY = 20000
BATCH_SIZE = 32
N_EPISODES = 5000
BEFORE_TRAIN = 10000
# annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_TIME = 20000
# Prioritized DQN configuration
PRIDQN_ENABLE = True
PRIDQN_CONFIG = {
    'epsilon': 0.01,              # small amount to avoid zero priority
    'alpha': 0.6,                 # [0~1] convert the importance of TD error to priority
    'beta': 0.4,                  # importance-sampling, from initial value increasing to 1
    'beta_increment_per_sampling': 0.001
}

# tensorflow Wrapper
def conv(in_tensor, out_channel, kernel_size, stide_size):
    biases_initializer = tf.constant_initializer(0.1)
    weights_initializer = tf.random_normal_initializer(mean=0, stddev = 0.0001) #tf.contrib.layers.xavier_initializer(uniform=True)
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
    return fc


class DeepQ():
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, N_ACTIONS, N_EPISODES, DISCOUNT):
        # init replay memory
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_CHANNEL = IMG_CHANNEL
        self.N_ACTIONS = N_ACTIONS
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        # self.replay_memory = deque(maxlen=REPLAY_MEMORY)
        self.replay_memory = Memory(capacity=REPLAY_MEMORY, enable_pri=PRIDQN_ENABLE, **PRIDQN_CONFIG)
        self.record = {'reward': [], 'survival_time': []}


    def approx_Q_network(self):
        x = tf.placeholder('float', [None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNEL])
        conv1 = conv(x, out_channel=32, kernel_size=8, stide_size=4)
        pool1 = maxpool(conv1, kernel_size=2, stide_size=2)
        conv2 = conv(pool1, out_channel=64, kernel_size=4, stide_size=2)
        pool2 = maxpool(conv2, kernel_size=2, stide_size=2)
        conv3 = conv(pool2, out_channel=64, kernel_size=3, stide_size=1)
        pool3 = maxpool(conv3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        fc1 = tf.nn.relu(fully_connect(flatten, 256))
        output_Q = tf.nn.softmax(fully_connect(fc1, self.N_ACTIONS))
        return x, output_Q

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.approx_Q_network()  # output_Q: (batch, N_ACTIONS)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])
        y = tf.placeholder("float", [None])
        if PRIDQN_ENABLE:
            ISWeights = tf.placeholder(tf.float32, [None, 1])
            abs_errors = tf.abs(y - max_action_Q)
            cost = tf.reduce_mean(ISWeights * tf.squared_difference(y, max_action_Q))
        else:
            cost = tf.reduce_mean(tf.squared_difference(y, max_action_Q))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        # Emulate and store trainsitions into replay_memory
        game = atari.Game('Breakout-v0')  #AirRaid-v0 #Enduro-v0
        state_t = game.initial_state()
        start_train_flag = False
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        init_op.run()

        for episode in range(self.N_EPISODES):
            t = 0
            terminal = False
            sum_reward = 0
            while not terminal:
                # Emulate and store trainsitions into replay_memory
                if(self.explore()):
                    action_t = game.random_action()
                    #print(action_t, end='\' ')
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1,80,80,4))})[0]
                    print(action_t, end=' ')
                for i in range(SKIP_FRAMES):
                    state_t1, reward_t, terminal, info = game.step(action_t)  # Execute the chosen action in emulator
                self.replay_memory.store([state_t, action_t, reward_t, state_t1, terminal])
                sum_reward += reward_t
                state_t = state_t1
                if terminal:
                    state_t = game.initial_state()
                t += SKIP_FRAMES
                # Train the approx_Q_network
                if len(self.replay_memory) >= BEFORE_TRAIN:
                    if not start_train_flag:
                        start_train_flag = True
                        print('------------------ Start Training ------------------')
                    transition_batch, batch_idx, weights = self.replay_memory.sample(BATCH_SIZE)
                    state_j, action_j, reward_j, state_j1, terminal_j1  = [], [], [], [], []
                    for transition in transition_batch:
                        state_j.append(transition[0])
                        action_j.append(transition[1])
                        reward_j.append(transition[2])
                        state_j1.append(transition[3])
                        terminal_j1.append(transition[4])
                    # the learned value for Q-learning
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * max_action_Q.eval(feed_dict={x: state_j1})[0] )
                    if PRIDQN_ENABLE:
                        _, errors, c = sess.run([train_step, abs_errors, cost],
                                                     feed_dict={x: state_j,
                                                                y: y_j,
                                                                ISWeights: weights})
                        for i in range(len(batch_idx)):  # update priority
                            idx = batch_idx[i]
                            self.replay_memory.update(idx, errors[i])
                    else:
                        _, c = sess.run([train_step, cost],
                                                     feed_dict={x: state_j,
                                                                y: y_j })
                    # train_step.run(feed_dict={x:state_j, y:y_j})

            self.record['reward'].append(sum_reward)
            self.record['survival_time'].append(t)
            print('\nEpisode {:3d}: sum of reward={:10.2f}, survival time ={:8d}'.format(episode, sum_reward, t))
            print('{:.2f} MB, replay memory size {:d}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, len(self.replay_memory)))
            self.global_time += t

        # Save model
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = saver.save(sess, '{}/model.ckpt'.format(directory))
        print('Model saved in file: %s' %save_path)
        with open('record/{}.json'.format(MODEL_ID), 'w') as f:
            json.dump(self.record, f, indent=1)

    def explore(self):
        if self.global_time <= SKIP_FRAMES*BEFORE_TRAIN:
            return True
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) < EXPLORE_TIME:
            self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / EXPLORE_TIME
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) ==  EXPLORE_TIME:
            print('------------------ Stop Annealing. Probability to explore = {:f} ------------------'.format(FINAL_EPSILON))
        return random.random() < self.epsilon

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(80, 80, 4, 6, N_EPISODES, DISCOUNT)
    dqn.train_network(sess)
