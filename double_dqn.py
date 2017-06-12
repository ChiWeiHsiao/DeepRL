# new
import tensorflow as tf
import numpy as np
from collections import deque 
import game_wrapper
import random
import resource
import os
import json

MODEL_ID = 'double-car'
directory = 'models/{}'.format(MODEL_ID)

# HyperParameter
HISTORY_LENGTH = 2
SKIP_FRAMES = 1 #4
DISCOUNT = 0.9 #0.99
LEARNING_RATE = 0.001
REPLAY_MEMORY = 3000
BATCH_SIZE = 32
N_EPISODES = 1000
BEFORE_TRAIN = 500
COPY_STEP = 4
# annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_TIME = 5000


# tensorflow Wrapper
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial) 

def bias_variable(shape):
    initial = tf.fill(shape, 0.01)
    return tf.Variable(initial)


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
        self.COPY_STEP = COPY_STEP
        self.EXPLORE_TIME = EXPLORE_TIME
        self.global_time = 0
        self.epsilon = INIT_EPSILON
        self.replay_memory = deque(maxlen=REPLAY_MEMORY)
        self.record = {'reward': [], 'time_used': []}
        self.human_transitions_file = human_transitions_file
        self.n_human_transitions = n_human_transitions
        if self.n_human_transitions > 0:
            self.load_human_transitions()
        self.W, self.b = self.initialize_weights()
        self.target_W, self.target_b = self.initialize_weights()

    def create_network(self, W, b):
        x = tf.placeholder('float', [None, self.N_HISTORY_LENGTH, self.N_OBSERVATIONS])
        flatten = tf.contrib.layers.flatten(x)
        fc1 = tf.nn.relu(tf.add(tf.matmul(flatten, W['f1']), b['f1'])) 
        output_Q = tf.nn.softmax(tf.add(tf.matmul(fc1, W['f2']), b['f2']))
        return x, output_Q

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.create_network(self.W, self.b)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])

        target_x, target_output_Q = self.create_network(self.target_W, self.target_b)
        target_max_action = tf.argmax(target_output_Q, axis=1)
        target_max_action_Q = tf.reduce_max(target_output_Q, reduction_indices=[1])

        y = tf.placeholder('float', [None])
        cost = tf.reduce_mean(tf.square(y - max_action_Q))
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
                self.store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal)
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
                    transition_batch = random.sample(self.replay_memory, BATCH_SIZE)
                    state_j, action_j, reward_j, state_j1, terminal_j1  = [], [], [], [], []
                    for transition in transition_batch:
                        state_j.append(transition[0])
                        action_j.append(transition[1])
                        reward_j.append(transition[2])
                        state_j1.append(transition[3])
                        terminal_j1.append(transition[4])
                    # the learned value for Q-learning
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * target_max_action_Q.eval(feed_dict={target_x: state_j1})[0])
                    train_step.run(feed_dict={x:state_j, y:y_j})
                    #this_run_cost = cost.eval(feed_dict={x:state_j, y:y_j})
                    #print('cost=',this_run_cost)
                    if t % self.COPY_STEP == 0:
                        self.copy_weights()
                
            self.record['reward'].append(sum_reward)
            self.record['time_used'].append(t)
            print('\nEpisode {:3d}: sum of reward={:10.2f}, time used={:8d}\n'.format(episode, sum_reward, t))
            #print('{:.2f} MB, replay memory size {:d}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, len(self.replay_memory)))
            self.global_time += t

        # Save model
        if not os.path.exists(directory):
            os.makedirs(directory)
        save_path = saver.save(sess, '{}/model.ckpt'.format(directory))
        print('Model saved in file: %s' %save_path)
        with open('record/{}.json'.format(MODEL_ID), 'w') as f:
            json.dump(self.record, f, indent=1)

    def test(self, sess, n_episodes=10):
        saver = tf.train.Saver()
        model_name = '%s/model' %directory
        saver.restore(sess, model_name)
        print('Model restored from {}'.format(model_name))
        x, output_Q = self.create_network()
        max_action = tf.argmax(output_Q, axis=1)

        state_t = game.initial_state()
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])

        self.game.env.render()
        self.game.initial_state()
        for episode in range(n_episodes):
            terminal = False
            total_reward = sum_reward = 0
            total_survival_time = survival_time = 0
            while not terminal:
                action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1,80,80,4))})[0]
                print(action_t, end='')
                state_t1, reward_t, terminal, info = game.step(action_t)  # Execute the chosen action in emulator
                sum_reward += reward_t
                state_t = state_t1
                if terminal:
                    state_t = self.game.initial_state()
                survival_time += 1
            print('Run {}: reward={:10.2f}, survival time ={:8d}'.format(episode, sum_reward, survival_time))
            total_reward += sum_reward
            total_survival_time += survival_time
        print('Average: reward={:10.2f}, survival time ={:8.2f}'.format(n_episodes, total_reward/n_episodes, total_survival_time/n_episodes))

    def explore(self):
        if self.global_time <= SKIP_FRAMES*BEFORE_TRAIN:
            return True
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) < self.EXPLORE_TIME:
            self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / self.EXPLORE_TIME
        elif (self.global_time - SKIP_FRAMES*BEFORE_TRAIN) ==  self.EXPLORE_TIME:
            print('------------------ Stop Annealing. Probability to explore = {:f} ------------------'.format(FINAL_EPSILON))
            self.EXPLORE_TIME -= 1
        return random.random() < self.epsilon

    def store_to_replay_memory(self, state_t, action_t, reward_t, state_t1, terminal):
        transition = [state_t, action_t, reward_t, state_t1, terminal]
        self.replay_memory.append(transition)
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()

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

    def initialize_weights(self):
        weight = {
            'f1': weight_variable([self.N_HISTORY_LENGTH*self.N_OBSERVATIONS, 20]),
            'f2': weight_variable([20, self.N_ACTIONS]),
        }
        bias = {
            'f1': bias_variable([20]),
            'f2': bias_variable([self.N_ACTIONS]),
        }
        return weight, bias

    def copy_weights(self):
        self.target_W['f1'].assign(self.W['f1'])
        self.target_W['f2'].assign(self.W['f2'])
        self.target_b['f1'].assign(self.b['f1'])
        self.target_b['f2'].assign(self.b['f2'])



if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(HISTORY_LENGTH, N_EPISODES, DISCOUNT, EXPLORE_TIME, 'MountainCar-v0', render=False,
             human_transitions_file='car_human_transitions.npz', n_human_transitions=0)#int(REPLAY_MEMORY*0.5))
    dqn.train_network(sess)

