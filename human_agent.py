import sys, gym
import time
import game_wrapper
from collections import deque
import numpy as np

game = game_wrapper.Game('MountainCar-v0', histoy_length=3, render=True)
game.env.seed(21)
ACTIONS = game.env.action_space.n
SKIP_CONTROL = 3
human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
outfile = 'human_agent_transitions/car_his4_skip4'
max_length = 10000

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def store_to_replay_memory(state_t, action_t, reward_t, state_t1, terminal):
    transition = [state_t, action_t, reward_t, state_t1, terminal]
    replay_memory.append(transition)
    if len(replay_memory) > REPLAY_MEMORY:
        replay_memory.popleft()

game.env.render()
game.env.unwrapped.viewer.window.on_key_press = key_press
game.env.unwrapped.viewer.window.on_key_release = key_release

def play():
    store_state_t, store_action_t, store_reward_t, store_state_t1, store_terminal = [], [], [], [], []
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    terminal = False
    state_t = game.initial_state()
    state_t1 = state_t = np.stack([state_t], axis=0)
    skip = 0
    t = 0
    while not (terminal and state_t1[0][0][0] > game.env.observation_space.high[0]-0.1):
        if not skip:
            action_t = human_agent_action
            skip = SKIP_CONTROL
            last_action = action_t
        else:
            action_t = last_action
            skip -= 1
        
        state_t1, reward_t, terminal, info = game.step(action_t)  # Execute the chosen action in emulator
        print(reward_t, end=', ')
        state_t1 = np.stack([state_t1], axis=0)
        if t == 0:
            store_state_t, store_action_t, store_reward_t, store_state_t1, store_terminal = state_t, [action_t], [reward_t], state_t1, [terminal]
        else:
            store_state_t = np.append(store_state_t, state_t, axis=0)
            store_action_t.append(action_t)
            store_reward_t.append(reward_t)
            store_state_t1 = np.append(store_state_t1, state_t1, axis=0)
            store_terminal.append(terminal)
        state_t = state_t1
        game.env.render()
        #time.sleep(0.05)

        if human_wants_restart: break
        while human_sets_pause:
            game.env.render()
            time.sleep(5)
        t = t + 1
    return store_state_t, np.array(store_action_t), np.array(store_reward_t), store_state_t1, np.array(store_terminal)

def store_human_transition(outfile, max_length):
    store_state_t, store_action_t, store_reward_t, store_state_t1, store_terminal = [], [], [], [], []
    first = True
    store_terminal = np.array([])
    while store_terminal.shape[0] < max_length:
        if first:
            store_state_t, store_action_t, store_reward_t, store_state_t1, store_terminal = play()
            first = False
        else:
            state_t, action_t, reward_t, state_t1, terminal = play()
            store_state_t = np.append(store_state_t, state_t, axis=0)
            store_action_t = np.append(store_action_t, action_t, axis=0)
            store_reward_t = np.append(store_reward_t, reward_t, axis=0)
            store_state_t1 = np.append(store_state_t1, state_t1, axis=0)
            store_terminal = np.append(store_terminal, terminal, axis=0)
        print('current length: {}'.format(store_state_t.shape))
    print('human transitions shape: {}, {}, {}, {}, {}'.format(store_state_t.shape, store_action_t.shape, store_reward_t.shape, store_state_t1.shape, store_terminal.shape))
    np.savez(outfile, state_t=store_state_t, action_t=store_action_t, reward_t=store_reward_t, state_t1=store_state_t1, terminal=store_terminal)
    print('transitions file save in {}.npz'.format(outfile))



if __name__ == '__main__':
    print("ACTIONS={}".format(ACTIONS))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")
    store_human_transition(outfile=outfile, max_length = max_length)

