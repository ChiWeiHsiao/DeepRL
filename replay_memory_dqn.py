'''
Deep Q Network with Experience Replay
NIPS 2013 Paper: https://arxiv.org/pdf/1312.5602.pdf)
'''
import tensorflow as tf
import numpy as np
from collections import deque
import random
import os
import json
import game_wrapper
from memory import Memory
try:
    import resource
except:
    pass

# Record & model filename to save
MODEL_ID = 'double-car'
directory = 'models/{}'.format(MODEL_ID)
# Specify game
GAME_NAME = 'MountainCar-v0'
RNEDER = False
N_EPISODES = 1000
# HyperParameter
HISTORY_LENGTH = 1
SKIP_FRAMES = 1 #4
DISCOUNT = 0.9 #0.99
LEARNING_RATE = 0.001
REPLAY_MEMORY = 3000
BEFORE_TRAIN = 500
BATCH_SIZE = 32
# Use human player transition or not
human_transitions_filename = 'car_human_transitions.npz'
n_human_transitions_used = 0 #int(REPLAY_MEMORY*0.5))
# Annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_TIME = 5000
# Prioritized DQN configuration
PRIDQN_ENABLE = True
PRIDQN_CONFIG = {
    'epsilon': 0.01,              # small amount to avoid zero priority
    'alpha': 0.6,                 # [0~1] convert the importance of TD error to priority
    'beta': 0.4,                  # importance-sampling, from initial value increasing to 1
    'beta_increment_per_sampling': 0.001
}

# tensorflow Wrapper
def fully_connect(in_tensor, n_out):
    biases_initializer = tf.constant_initializer(0.01)
    weights_initializer = tf.random_normal_initializer(mean=0, stddev = 0.1)
    fc = tf.contrib.layers.fully_connected(in_tensor, n_out, activation_fn=None, biases_initializer=biases_initializer, weights_initializer=weights_initializer, trainable=True)
    return fc

class DeepQ():
    def __init__(self, N_HISTORY_LENGTH, N_EPISODES, DISCOUNT, EXPLORE_TIME, game_name, render=False, human_transitions_file=None, n_human_transitions=0):
        self.game_name = game_name
        self.render = render
        self.N_HISTORY_LENGTH = N_HISTORY_LENGTH
        self.game = game_wrapper.Game(self.game_name, self.N_HISTORY_LENGTH, self.render)
        self.N_OBSERVATIONS = len(self.game.env.observation_space.high)
        self.N_ACTIONS = self.game.env.action_space.n
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.EXPLORE_TIME = EXPLORE_TIME
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        # self.replay_memory = deque(maxlen=REPLAY_MEMORY)
        self.record = {'reward': [], 'time_used': []}
        self.human_transitions_file = human_transitions_file
        self.n_human_transitions = n_human_transitions
        if self.n_human_transitions > 0:
            self.load_human_transitions()
        self.replay_memory = Memory(capacity=REPLAY_MEMORY, enable_pri=PRIDQN_ENABLE, **PRIDQN_CONFIG)

    def approx_Q_network(self):
        x = tf.placeholder('float', [None, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS])
        flatten = tf.contrib.layers.flatten(x)
        fc1 = tf.nn.relu(fully_connect(flatten, 20))
        output_Q = tf.nn.softmax(fully_connect(fc1, self.N_ACTIONS))
        return x, output_Q

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.approx_Q_network()  # output_Q: (batch, N_ACTIONS)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])
        y = tf.placeholder('float', [None])
        if PRIDQN_ENABLE:
            ISWeights = tf.placeholder(tf.float32, [None, 1])
            abs_errors = tf.abs(y - max_action_Q)
            cost = tf.reduce_mean(ISWeights * tf.squared_difference(y, max_action_Q))
        else:
            cost = tf.reduce_mean(tf.squared_difference(y, max_action_Q))
        train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cost)

        # Emulate and store trainsitions into replay_memory
        state_t = self.game.initial_state()
        start_train_flag = False
        saver = tf.train.Saver()
        #saver.restore(sess, 'models/{}/model.ckpt'.format(MODEL_ID))
        #print('model restore.')
        init_op = tf.global_variables_initializer()
        init_op.run()

        for episode in range(self.N_EPISODES):
            t = 0
            terminal = False
            sum_reward = 0
            while not terminal:
                # Emulate and store trainsitions into replay_memory
                if(self.explore()):
                    action_t = self.game.random_action()
                    #print(action_t, end='\' ')
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS))})[0]
                    #print(action_t, end=' ')
                for i in range(SKIP_FRAMES):
                    state_t1, reward_t, terminal, info = self.game.step(action_t)  # Execute the chosen action in emulator

                self.replay_memory.store([state_t, action_t, reward_t, state_t1, terminal])
                # self.store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal)
                sum_reward += reward_t
                state_t = state_t1
                if terminal:
                    state_t = self.game.initial_state()
                t += SKIP_FRAMES
                # Train the approx_Q_network
                if len(self.replay_memory) >= BEFORE_TRAIN:
                    if not start_train_flag:
                        start_train_flag = True
                        print('------------------ Start Training ------------------')
                    # transition_batch = random.sample(self.replay_memory, BATCH_SIZE)
                    transition_batch, batch_idx, weights = self.replay_memory.sample(BATCH_SIZE)

                    state_j, action_j, reward_j, state_j1, terminal_j1  = [], [], [], [], []
                    for transition in transition_batch:
                        state_j.append(transition[0])
                        action_j.append(transition[1])
                        reward_j.append(transition[2])
                        state_j1.append(transition[3])
                        terminal_j1.append(transition[4])

                    # the learned value for Q-learning
                    # print(state_j1)
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * max_action_Q.eval(feed_dict={x: state_j1})[0] )
                    #this_run_cost = cost.eval(feed_dict={x:state_j, y:y_j})
                    #print('cost=',this_run_cost)

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

            self.record['reward'].append(sum_reward)
            self.record['time_used'].append(t)
            print('\nEpisode {:3d}: sum of reward={:10.2f}, time used={:8d}'.format(episode, sum_reward, t))
            #print('{:.2f} MB, replay memory size {:d}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, len(self.replay_memory)))
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
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) < self.EXPLORE_TIME:
            self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / self.EXPLORE_TIME
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) ==  self.EXPLORE_TIME:
            print('------------------ Stop Annealing. Probability to explore = {:f} ------------------'.format(FINAL_EPSILON))
            self.EXPLORE_TIME -= 1
        return random.random() < self.epsilon

    def load_human_transitions(self):
        data = np.load(self.human_transitions_file)
        print('There are {} human transitions available, and {} are used'.format(data['state_t'].shape[0], self.n_human_transitions))
        state_t = data['state_t']
        action_t = data['action_t']
        reward_t = data['reward_t']
        state_t1 = data['state_t1']
        terminal = data['terminal']
        for i in range(self.n_human_transitions):
            transition = [state_t[i], action_t[i], reward_t[i], state_t1[i], terminal[i]]
            self.replay_memory.append(transition)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(HISTORY_LENGTH, N_EPISODES, DISCOUNT, EXPLORE_TIME, GAME_NAME, render=RNEDER,
             human_transitions_file=human_transitions_filename, n_human_transitions=n_human_transitions_used)
    dqn.train_network(sess)
