'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque 
import game_wrapper
import random
import resource
import os
import json

MODEL_ID = 'replay-1'
directory = 'models/{}'.format(MODEL_ID)

# HyperParameter
SKIP_FRAMES = 1 #4
DISCOUNT = 0.99
LEARNING_RATE = 0.0001
REPLAY_MEMORY = 2000
BATCH_SIZE = 32
N_EPISODES = 1000
BEFORE_TRAIN = 1000
# annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_TIME = 2000


# tensorflow Wrapper
def fully_connect(in_tensor, n_out):
    biases_initializer = tf.constant_initializer(0.01)
    weights_initializer = tf.truncated_normal_initializer(mean=0, stddev = 0.01)
    fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
    return fc


class DeepQ():
    def __init__(self, N_OBSERVATIONS, N_HISTORY_LENGTH, N_ACTIONS, N_EPISODES, DISCOUNT, render):
        # init replay memory
        self.N_OBSERVATIONS = N_OBSERVATIONS
        self.N_HISTORY_LENGTH = N_HISTORY_LENGTH
        self.N_ACTIONS = N_ACTIONS
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.render = render
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        self.replay_memory = deque(maxlen=REPLAY_MEMORY)
        self.record = {'reward': [], 'time_used': []}


    def approx_Q_network(self):
        x = tf.placeholder('float', [None, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS])
        flatten = tf.contrib.layers.flatten(x)
        fc1 = tf.nn.relu(fully_connect(flatten, 128))
        fc2 = tf.nn.relu(fully_connect(fc1, 64))
        output_Q = tf.nn.softmax(fully_connect(fc2, self.N_ACTIONS))
        return x, output_Q

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.approx_Q_network()  # output_Q: (batch, N_ACTIONS)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])
        y = tf.placeholder("float", [None])
        cost = tf.reduce_mean(tf.square(y - max_action_Q))
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        # Emulate and store trainsitions into replay_memory
        game = game_wrapper.Game('MountainCar-v0', self.render)
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
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS))})[0]
                    print(action_t, end=' ')
                for i in range(SKIP_FRAMES):
                    state_t1, reward_t, terminal, info = game.step(action_t)  # Execute the chosen action in emulator
                self.store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal)
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
                    transition_batch = random.sample(self.replay_memory, BATCH_SIZE)
                    state_j, action_j, reward_j, state_j1, terminal_j1  = [], [], [], [], []
                    for transition in transition_batch:
                        state_j.append(transition[0])
                        action_j.append(transition[1])
                        reward_j.append(transition[2])
                        state_j1.append(transition[3])
                        terminal_j1.append(transition[4])
                    # the learned value for Q-learning
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * max_action_Q.eval(feed_dict={x: state_j1})[0] )
                    train_step.run(feed_dict={x:state_j, y:y_j})
                
            self.record['reward'].append(sum_reward)
            self.record['time_used'].append(t)
            print('\nEpisode {:3d}: sum of reward={:10.2f}, time used={:8d}'.format(episode, sum_reward, t))
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

    def store_to_replay_memory(self, state_t, action_t, reward_t, state_t1, terminal):
        transition = [state_t, action_t, reward_t, state_t1, terminal]
        self.replay_memory.append(transition)
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(2, 2, 3, N_EPISODES, DISCOUNT, False)
    dqn.train_network(sess)

