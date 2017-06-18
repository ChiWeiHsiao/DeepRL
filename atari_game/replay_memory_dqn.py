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

random.seed(1)
tf.set_random_seed(1)
np.random.seed(1)
# Record & model filename to save
MODEL_ID = 'replay-rhist4-skip4-dis97-lr1e4-eps1e5-ne1000'
print(MODEL_ID)
directory = 'models/{}'.format(MODEL_ID)
# Specify game
GAME_NAME = 'Pong-v0'
RENDER = False
N_EPISODES = 10
MAX_STEPS = 15000
# HyperParameter
HISTORY_LENGTH = 4
SKIP_FRAMES = 4
DISCOUNT = 0.97
LEARNING_RATE = 1e-4
REPLAY_MEMORY = 10000
BEFORE_TRAIN = 10000
BATCH_SIZE = 32
N_HIDDEN_NODES = 40
# Use human player transition or not
human_transitions_filename = 'human_agent_transitions/car_history1.npz'
n_human_transitions_used = 0
# Annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EPSILON_DECREMENT = 1e-5
# Prioritized DQN configuration
PRIDQN_ENABLE = False
PRIDQN_CONFIG = {
    'epsilon': 0.01,              # small amount to avoid zero priority
    'alpha': 0.6,                 # [0~1] convert the importance of TD error to priority
    'beta': 0.4,                  # importance-sampling, from initial value increasing to 1
    'beta_increment_per_sampling': 0.001
}
# tensorflow Wrapper
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.3)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.fill(shape, 0.1)
    return tf.Variable(initial)

def conv(x, W, b, stride):
    stride_list = [1, stride, stride, 1]
    conv = tf.nn.conv2d(x, W, stride_list, 'SAME') + b
    return tf.nn.relu(conv)

def maxpool(in_tensor, kernel_size, stide_size):
    pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=kernel_size, stride=stide_size, padding='SAME')
    return pool


class DeepQ():
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, N_HISTORY_LENGTH, N_EPISODES, DISCOUNT, EPSILON_DECREMENT, game_name, render=False, human_transitions_file=None, n_human_transitions=0):
        self.game_name = game_name
        self.render = render
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.N_HISTORY_LENGTH = N_HISTORY_LENGTH
        self.game = atari.Game(self.game_name, self.N_HISTORY_LENGTH, self.render)
        #self.game = atari.Game('AirRaid-v0')
        self.game.env.seed(21)
        self.N_OBSERVATIONS = len(self.game.env.observation_space.high)
        self.N_ACTIONS = self.game.env.action_space.n
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        self.EPSILON_DECREMENT = EPSILON_DECREMENT
        # self.replay_memory = deque(maxlen=REPLAY_MEMORY)
        self.record = {'reward': [], 'time_used': [], 'cost': []}
        self.testing_record = {'reward': [], 'time_used': []}
        self.W, self.b = self.initialize_weights()
        self.replay_memory = Memory(capacity=REPLAY_MEMORY, enable_pri=PRIDQN_ENABLE, **PRIDQN_CONFIG)
        self.human_transitions_file = human_transitions_file
        self.n_human_transitions = n_human_transitions
        if self.n_human_transitions > 0:
            self.load_human_transitions()

    def create_network(self, W, b):
        x = tf.placeholder('float', [None, self.IMG_WIDTH, self.IMG_HEIGHT, self.N_HISTORY_LENGTH])
        h_c1 = conv(x, self.W['c1'], self.b['c1'], 4)
        pool1 = maxpool(h_c1, kernel_size=2, stide_size=2)
        h_c2 = conv(pool1, self.W['c2'], self.b['c2'], 2)
        pool2 = maxpool(h_c2, kernel_size=2, stide_size=2)
        h_c3 = conv(pool2, self.W['c3'], self.b['c3'], 1)
        pool3 = maxpool(h_c3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        h_f1 = tf.nn.relu(tf.add(tf.matmul(flatten, self.W['f1']), self.b['f1']))
        output_Q = tf.add(tf.matmul(h_f1, self.W['f2']), self.b['f2'])
        #output_Q = tf.nn.relu(output_Q)
        return x, output_Q        

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.create_network(self.W, self.b)
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
            t = sum_reward = sum_cost = 0
            terminal = False
            state_t = self.game.initial_state()
            while not (terminal or t >= MAX_STEPS):
                # Emulate and store trainsitions into replay_memory
                if(self.explore()):
                    action_t = self.game.random_action()
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS))})[0]
                # Repeat the selected action for SKIP_FRAMES steps
                for i in range(SKIP_FRAMES):
                    state_t1, reward_t, terminal, info = self.game.step(action_t)  # Execute the chosen action in emulator
                    self.replay_memory.store([state_t, action_t, reward_t, state_t1, terminal])
                    sum_reward += reward_t
                    t += 1
                    state_t = state_t1
                    if terminal or t >= MAX_STEPS:
                        break
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
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * max_action_Q.eval(feed_dict={x: state_j1})[0])


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
                    sum_cost += c

            self.record['reward'].append(sum_reward)
            self.record['time_used'].append(t)
            self.record['cost'].append(sum_cost/t)
            print('Episode {:3d}: sum of reward={:10.2f}, time used={:8d}'.format(episode+1, sum_reward, t))
            print('current explore={:.5f}'.format(self.epsilon))
            print('avg cost = {}\n'.format(sum_cost/t))
            #print('{:.2f} MB, replay memory size {:d}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, len(self.replay_memory)))
            self.global_time += t

            if episode % 500 == 0:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = saver.save(sess, '{}/model.ckpt'.format(directory))
                print('Model saved in file: %s' %save_path)
                with open('record/{}.json'.format(MODEL_ID), 'w') as f:
                    json.dump(self.record, f, indent=1) 
        # Save model
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = saver.save(sess, '{}/model.ckpt'.format(directory))
        print('Model saved in file: %s' %save_path)
        with open('record/{}.json'.format(MODEL_ID), 'w') as f:
            json.dump(self.record, f, indent=1)


    def explore(self):
        if len(self.replay_memory) < BEFORE_TRAIN:
            return True
        elif self.epsilon > FINAL_EPSILON:
            self.epsilon -= self.EPSILON_DECREMENT
        return random.random() < self.epsilon

    def test(self, sess):
        self.epsilon = 0.1 #0
        saver = tf.train.Saver()
        model_name = '%s/model.ckpt' % directory
        saver.restore(sess, model_name)
        print('Model restored from {}'.format(model_name))
        x, output_Q = self.create_network(self.W, self.b)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])
        n_episodes = 50
        total_reward = total_time = 0
        for episode in range(n_episodes):
            self.game.env.render()
            terminal = False
            sum_reward = 0
            t = 0
            state_t = self.game.initial_state()
            while not (terminal or t >= MAX_STEPS):
                if(self.explore()):
                    action_t = self.game.random_action()
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1, self.IMG_WIDTH, self.IMG_HEIGHT, self.N_HISTORY_LENGTH))})[0]
                # Repeat the selected action for SKIP_FRAMES steps
                for i in range(SKIP_FRAMES):
                    state_t1, reward_t, terminal, info = self.game.step(action_t)  # Execute the chosen action in emulator
                    self.replay_memory.store([state_t, action_t, reward_t, state_t1, terminal])
                    sum_reward += reward_t
                    t += 1
                    state_t = state_t1
                    self.game.env.render()
                    if terminal or t >= MAX_STEPS:
                        break
            print('Test {}: reward={:5.2f}, time ={:3d}'.format(episode, sum_reward, t))
            self.testing_record['reward'].append(sum_reward)
            self.testing_record['time_used'].append(t)
            total_reward += sum_reward
            total_time += t
        print('Average: reward={:5.2f}, time ={:3.2f}'.format(total_reward/n_episodes, total_time/n_episodes))
        with open('record/{}_test.json'.format(MODEL_ID), 'w') as f:
          json.dump(self.testing_record, f, indent=1)

    def load_human_transitions(self):
        data = np.load(self.human_transitions_file)
        print('There are {} human transitions available, and {} are used'.format(data['state_t'].shape[0], self.n_human_transitions))
        state_t = data['state_t']
        action_t = data['action_t']
        reward_t = data['reward_t']
        state_t1 = data['state_t1']
        terminal = data['terminal']
        for i in range(self.n_human_transitions):
            self.replay_memory.store([state_t[i], action_t[i], reward_t[i], state_t1[i], terminal[i]])

    def initialize_weights(self):
        weight = {
            'c1': weight_variable([8,8,4,32]),
            'c2': weight_variable([4,4,32,64]),
            'c3': weight_variable([3,3,64,64]),
            'f1': weight_variable([256, 256]),
            'f2': weight_variable([256, self.N_ACTIONS]),
        }
        bias = {
            'c1': bias_variable([32]),
            'c2': bias_variable([64]),
            'c3': bias_variable([64]),
            'f1': bias_variable([256]),
            'f2': bias_variable([self.N_ACTIONS]),
        }
        return weight, bias



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(80, 80, HISTORY_LENGTH, N_EPISODES, DISCOUNT, EPSILON_DECREMENT, GAME_NAME, render=RENDER,
             human_transitions_file=human_transitions_filename, n_human_transitions=n_human_transitions_used)
    dqn.train_network(sess)
    dqn.test(sess)

