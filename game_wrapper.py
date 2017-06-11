import gym
from gym import wrappers
import numpy as np
from PIL import Image
from collections import deque


class Game():
    def __init__(self, game_name, histoy_length, render=False):
        self.env =  gym.make(game_name)
        #self.env = wrappers.Monitor(self.env, 'records/atari-experiment-1')
        self.render = render
        self.n_actions = self.env.action_space.n
        self.n_observation = len(self.env.observation_space.high)
        self.histoy_length = histoy_length  # One state contains 'histoy_length' observations
        self.state_buffer = deque() # Buffer keep 'histoy_length-1' observations

    def initial_state(self):
        ''' Initialize game. Prepare initial state and state_buffer
        Reset the game
        Build initial state with 'histoy_length' * first observation.
        Prepare state_buffer with 'histoy_length-1' * first observation
        '''
        self.state_buffer = deque()  # Clear the state buffer
        observation = self.env.reset()
        if self.render:
            self.env.render()
        observation_reshaped = np.stack([observation], axis=0)
        state = observation_reshaped
        for i in range(self.histoy_length-1):
            state = np.concatenate((state, observation_reshaped), axis=0) 
        # Prepare 'histoy_length-1' observations in buffer
        for i in range(self.histoy_length-1):
            self.state_buffer.append(observation)
        return state

    def step(self, action):
        ''' Execute the given action, move to next time point
        Execute one action in the game. Transition from state_t to state_t1, with immediate reward.
        Build current state ( = previous 'histoy_length-1' observations + current observation ).
        Pop out the oldest observation, push latest observation into state_buffer.
        '''
        if self.render:
            self.env.render()
        observation_t1, reward, terminal_t1, info = self.env.step(action)
        reward = abs(observation_t1[0] - (-0.5))
        previous_observations = np.array(self.state_buffer)
        state_t1 = np.empty((self.histoy_length, self.n_observation))
        for i in range(previous_observations.shape[0]):
            state_t1[i, ...] = previous_observations[i]
        state_t1[self.histoy_length-1, ...] = observation_t1
        # Pop the oldest observation, add the current observation to the queue
        if len(self.state_buffer) >= 1:
            self.state_buffer.popleft()
        self.state_buffer.append(observation_t1)

        return state_t1, reward, terminal_t1, info


    def random_action(self):
        return self.env.action_space.sample()

    def show_game_info(self):
        ''' Show information about the game'''
        print('Action space: {}'.format(self.env.action_space))
        print('Observation space: {}'.format(self.env.observation_space))


def example():
    game = Game('MountainCar-v0')
    game.show_game_info()
    init_state = game.initial_state()
    game_over = False
    N_EPISODES = 3
    for i_episode in range(N_EPISODES):
        while not game_over:
            game.env.render()
            action = game.env.action_space.sample()
            state, reward, game_over, info = game.step(action)
            print('state:', state.shape)
        game.env.reset()


if __name__ == '__main__':
    example()


