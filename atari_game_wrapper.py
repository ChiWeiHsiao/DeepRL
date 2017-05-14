import gym
from gym import wrappers
import numpy as np
from PIL import Image
from collections import deque
from random import randint


class Game():
    def __init__(self, game_name):
        self.env =  gym.make(game_name)
        #self.env = wrappers.Monitor(self.env, '/tmp/cartpole-experiment-1')
        self.n_actions = self.env.action_space.n
        self.resize_width = 80
        self.resize_height = 80
        self.histoy_length = 4  # One state contains 'histoy_length' frames
        self.state_buffer = deque() # Buffer keep 'histoy_length-1' frames

    def initial_state(self):
        ''' Initialize game. Prepare initial state and state_buffer
        Reset the game
        Build initial state with 'histoy_length' * first observation.
        Prepare state_buffer with 'histoy_length-1' * first observation
        '''
        self.state_buffer = deque()  # Clear the state buffer
        observation = self.env.reset()
        frame = self.preprocess_frame(observation)
        single_frame = np.reshape(frame, (self.resize_width, self.resize_height, 1))
        state = single_frame
        for i in range(self.histoy_length-1):
            state = np.concatenate((state, single_frame), axis=-1) 
        assert state.shape == (self.resize_width, self.resize_height, self.histoy_length)
        # Prepare 'histoy_length-1' frames in buffer
        for i in range(self.histoy_length-1):
            self.state_buffer.append(frame)
        return state

    def step(self, action):
        ''' Execute the given action, move to next time point
        Execute one action in the game. Transition from state_t to state_t1, with immediate reward.
        Build current state ( = previous 'histoy_length-1' frames + current frame ).
        Pop out the oldest frame, push latest frame into state_buffer.
        '''
        observation_t1, reward, terminal_t1, info = self.env.step(action)
        observation_t1 = self.preprocess_frame(observation_t1)

        previous_frames = np.array(self.state_buffer) #(3, 80, 80)
        state_t1 = np.empty((self.resize_width, self.resize_height, self.histoy_length))
        for i in range(previous_frames.shape[0]):
            state_t1[..., i] = previous_frames[i]
        state_t1[..., self.histoy_length-1] = observation_t1
        assert state_t1.shape == (self.resize_width, self.resize_height, self.histoy_length)
        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(observation_t1)

        return state_t1, reward, terminal_t1, info

    def preprocess_frame(self, observation):
        ''' Preprocess screen image data
        Color is not important here, so change RGB to gray scale.
        Resize the smaller image for training efficiency.
        '''
        gray_img = np.dot(observation[...,:3], [0.299, 0.587, 0.114])
        gray_img = Image.fromarray(gray_img)
        resized_img = gray_img.resize((self.resize_width , self.resize_height), Image.BILINEAR )
        #result = np.reshape(resized_img, (self.resize_width , self.resize_height, 1))
        result = np.reshape(resized_img, (self.resize_width , self.resize_height))
        return result

    def show_game_info(self):
        ''' Show information about the game'''
        print('Action space: {}'.format(self.env.action_space)) # Discrete(6) => 6 actions,  either 0 or 1
        print('Observation space: {}'.format(self.env.observation_space)) # Box(250, 160, 3), rgb

    def random_action(self):
        return randint(0, self.n_actions-1)

def example():
    game = Game('AirRaid-v0')
    game.show_game_info()
    init_state = game.initial_state()
    game_over = False
    N_EPISODES = 3
    for i_episode in range(N_EPISODES):
        while not game_over:
            game.env.render()
            action = game.env.action_space.sample()
            state, reward, game_over, info = game.step(action)
        game.env.reset()


if __name__ == '__main__':
    example()


