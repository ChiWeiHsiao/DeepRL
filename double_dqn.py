import tensorflow as tf
import numpy as np
from collections import deque 
import atari_game_wrapper as atari
import random
import resource

MODEL_ID = 'double'
RESTORE_MODEL = False
RESTORE_GLOBAL_TIME = 0
# If this is the first time of running this program, 
# set 'RESTORE_MODEL = False' and 'RESTORE_GLOBAL_TIME = 0'.
# Otherwise, set 'RESTORE_MODEL = True' and 'RESTORE_GLOBAL_TIME = THE_TIME_YOU_WANT_TO_RESTORE'
RESTORE_MODEL_NAME = 'models/{}-{}'.format(MODEL_ID, RESTORE_GLOBAL_TIME) # The filename of trained model you want to restore

# HyperParameter
DISCOUNT = 0.99
REPLAY_MEMORY = 15000
BATCH_SIZE = 32
N_EPISODES = 100
BEFORE_TRAIN = 10000
# annealing for exploration probability
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_TIME = 10000


# tensorflow Wrapper
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial) 

def bias_variable(shape):
    initial = tf.fill(shape, 0.01)
    return tf.Variable(initial)

def conv(x, W, b, stride):
    stride_list = [1, stride, stride, 1]
    conv = tf.nn.conv2d(x, W, stride_list, 'SAME') + b
    return tf.nn.relu(conv)

def maxpool(in_tensor, kernel_size, stide_size):
    pool = tf.contrib.layers.max_pool2d(in_tensor, kernel_size=kernel_size, stride=stide_size, padding='SAME')
    return pool


class DeepQ():
    def __init__(self, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, N_ACTIONS, N_EPISODES, DISCOUNT):
        # init replay memory
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_CHANNEL = IMG_CHANNEL
        self.N_ACTIONS = N_ACTIONS
        self.N_EPISODES = N_EPISODES
        self.DISCOUNT = DISCOUNT
        self.COPY_STEP = 4
        self.W, self.b = self.initialize_weights()
        self.target_W, self.target_b = self.initialize_weights()
        
        if not  RESTORE_MODEL:
            self.global_time = 0
            self.epsilon = INIT_EPSILON
        else:
            self.global_time = RESTORE_GLOBAL_TIME
            if ( self.global_time - BEFORE_TRAIN ) < EXPLORE_TIME:
                self.epsilon = INIT_EPSILON - (self.global_time - BEFORE_TRAIN) * (INIT_EPSILON - FINAL_EPSILON) / EXPLORE_TIME
            else:
                self.epsilon = FINAL_EPSILON
        self.replay_memory = deque(maxlen=REPLAY_MEMORY)

    def approx_Q_network(self):
        x = tf.placeholder('float', [None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNEL])
        h_c1 = conv(x, self.W['c1'], self.b['c1'], 4)
        pool1 = maxpool(h_c1, kernel_size=2, stide_size=2)
        h_c2 = conv(pool1, self.W['c2'], self.b['c2'], 2)
        pool2 = maxpool(h_c2, kernel_size=2, stide_size=2)
        h_c3 = conv(pool2, self.W['c3'], self.b['c3'], 1)
        pool3 = maxpool(h_c3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        h_f1 = tf.nn.relu(tf.add(tf.matmul(flatten, self.W['f1']), self.b['f1']))
        output_Q = tf.nn.relu(tf.add(tf.matmul(h_f1, self.W['f2']), self.b['f2']))
        return x, output_Q

    def target_Q_network(self):
        x = tf.placeholder('float', [None, self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNEL])
        h_c1 = conv(x, self.target_W['c1'], self.target_b['c1'], 4)
        pool1 = maxpool(h_c1, kernel_size=2, stide_size=2)
        h_c2 = conv(pool1, self.target_W['c2'], self.target_b['c2'], 2)
        pool2 = maxpool(h_c2, kernel_size=2, stide_size=2)
        h_c3 = conv(pool2, self.target_W['c3'], self.target_b['c3'], 1)
        pool3 = maxpool(h_c3, kernel_size=2, stide_size=2)
        flatten = tf.contrib.layers.flatten(pool3)
        h_f1 = tf.nn.relu(tf.add(tf.matmul(flatten, self.target_W['f1']), self.target_b['f1']))
        target_Q = tf.nn.relu(tf.add(tf.matmul(h_f1, self.target_W['f2']), self.target_b['f2']))
        return x, target_Q

    def train_network(self, sess):
        # Define cost function of network
        x, output_Q = self.approx_Q_network()  # output_Q: (batch, N_ACTIONS)
        max_action = tf.argmax(output_Q, axis=1)
        max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])

        target_x, target_output_Q = self.target_Q_network()
        target_max_action = tf.argmax(target_output_Q, axis=1)
        target_max_action_Q = tf.reduce_max(output_Q, reduction_indices=[1])

        y = tf.placeholder("float", [None])
        cost = tf.reduce_mean(tf.square(y - max_action_Q))
        train_step = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(cost)

        # Emulate and store trainsitions into replay_memory
        game = atari.Game('AirRaid-v0')
        state_t = game.initial_state()
        start_train_flag = False
        saver = tf.train.Saver()
        if not RESTORE_MODEL:
            init_op = tf.global_variables_initializer()
            init_op.run()
        else:
            saver.restore(sess, RESTORE_MODEL_NAME)
            print('Model restored. Restored global time = {}'.format(RESTORE_GLOBAL_TIME))

        for episode in range(self.N_EPISODES):
            t = 0
            terminal = False
            sum_reward = 0
            while not terminal:
                # Emulate and store trainsitions into replay_memory
                if(self.explore()):
                    action_t = game.random_action()
                else:
                    action_t = max_action.eval(feed_dict={x: np.reshape(state_t, (1,80,80,4))})[0]
                state_t1, reward_t, terminal, info = game.step(action_t)  # Execute the chosen action in emulator
                self.store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal)
                sum_reward += reward_t
                state_t = state_t1
                if terminal:
                    state_t = game.initial_state()
                self.global_time += 1
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
                    y_j = np.where(terminal_j1, reward_j, reward_j + self.DISCOUNT * target_max_action_Q.eval(feed_dict={x: state_j1})[0] )
                    train_step.run(feed_dict={x:state_j, y:y_j})

                    if self.global_time % self.COPY_STEP == 0:
                        self.copy_weights()

                #if self.global_time > BEFORE_TRAIN and self.global_time % 10000 == 0:
                   #saver.save(sess, 'models/' + MODEL_ID, global_step = self.global_time)
                   #print('\tSave model as "models/{}-{}"'.format(MODEL_ID, self.global_time))
                
            print('Episode {:3d}: sum of reward={:10.2f}, survival time ={:8d}'.format(episode, sum_reward, self.global_time-RESTORE_GLOBAL_TIME))
            print('{:.2f} MB, replay memory size {:d}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, len(self.replay_memory)))
        saver.save(sess, 'models/' + 'final_' + MODEL_ID, global_step = self.global_time)
        print('\tFinal model as "models/{}-{}"'.format(MODEL_ID, self.global_time))


    def explore(self):
        if self.global_time <= BEFORE_TRAIN:
            return True
        elif (self.global_time - BEFORE_TRAIN) < EXPLORE_TIME:
            self.epsilon -= (INIT_EPSILON - FINAL_EPSILON) / EXPLORE_TIME
        elif (self.global_time - BEFORE_TRAIN) ==  EXPLORE_TIME:
            print('------------------ Stop Annealing. Probability to explore = {:f} ------------------'.format(FINAL_EPSILON))
        return random.random() < self.epsilon

    def store_to_replay_memory(self, state_t, action_t, reward_t, state_t1, terminal):
        transition = [state_t, action_t, reward_t, state_t1, terminal]
        self.replay_memory.append(transition)
        if len(self.replay_memory) > REPLAY_MEMORY:
            self.replay_memory.popleft()

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

    def copy_weights(self):
        self.target_W['c1'].assign(self.W['c1'])
        self.target_W['c2'].assign(self.W['c2'])
        self.target_W['c3'].assign(self.W['c3'])
        self.target_W['f1'].assign(self.W['f1'])
        self.target_W['f2'].assign(self.W['f2'])

        self.target_b['c1'].assign(self.b['c1'])
        self.target_b['c2'].assign(self.b['c2'])
        self.target_b['c3'].assign(self.b['c3'])
        self.target_b['f1'].assign(self.b['f1'])
        self.target_b['f2'].assign(self.b['f2'])


if __name__ == '__main__':
    sess = tf.InteractiveSession()
    dqn = DeepQ(80, 80, 4, 6, N_EPISODES, DISCOUNT)
    dqn.train_network(sess)

